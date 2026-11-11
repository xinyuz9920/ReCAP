"""Target-data partition and nested residual-coverage candidate construction.

Only normalized image/visible-QA vectors and owner metadata enter this module.
No classifier, source-domain partitions, answer likelihoods, or reference subsets.
"""
import time

import numpy as np
from scipy import sparse

from .greedy import lazy_greedy


def normalized(array, allow_zero=False):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError('Expected finite two-dimensional vectors')
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not allow_zero and np.any(norms < 1e-12):
        raise ValueError('Zero feature vector')
    return array / np.maximum(norms, 1e-12)


def record_features(image, text, owner, turns=None):
    image = normalized(image)
    text = normalized(text)
    owner = np.asarray(owner, dtype=np.int64)
    counts = np.bincount(owner, minlength=len(image))
    if len(counts) != len(image) or np.any(counts <= 0):
        raise ValueError('Every parent record must have visible supervised QA')
    if turns is not None and not np.array_equal(counts, turns):
        raise ValueError('Visible QA owner/turn counts differ')
    rows = np.arange(len(owner))
    pool = sparse.csr_matrix((1/counts[owner], (owner, rows)), shape=(len(image), len(owner)))
    mean_text = normalized(pool @ text, allow_zero=True)
    return normalized(np.concatenate([image, mean_text], axis=1))


def balanced_pca_groups(record_features, depth=7, seed=42):
    """Deterministic-for-fixed-seed median-split approximate PCA tree.

    Use 24 power iterations initialized by a seeded dense vector at each node.
    Degenerate nodes and equal projections are split by original row index.
    """
    started = time.monotonic()
    z = np.asarray(record_features, dtype=np.float32)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError('Invalid partition vectors')
    if depth < 0 or 2**depth > len(z):
        raise ValueError('Partition depth exceeds parent capacity')
    labels = np.empty(len(z), np.int32)
    nodes = []
    leaf_sizes = []

    def split(ids, level, leaf_prefix):
        if level == depth:
            labels[ids] = leaf_prefix
            leaf_sizes.append(len(ids))
            return
        values = z[ids].astype(np.float64)
        centered = values-values.mean(axis=0, keepdims=True)
        variance = np.einsum('ij,ij->j', centered, centered)
        generator = np.random.default_rng(np.random.SeedSequence([int(seed), level, leaf_prefix]))
        axis = generator.normal(size=z.shape[1])
        axis /= np.linalg.norm(axis)
        degenerate = float(variance.max()) < 1e-20
        if not degenerate:
            for _ in range(24):
                updated = centered.T @ (centered @ axis)
                norm = np.linalg.norm(updated)
                if norm < 1e-20:
                    degenerate = True
                    break
                axis = updated/norm
            if axis[int(np.argmax(np.abs(axis)))] < 0:
                axis = -axis
        scores = np.zeros(len(ids)) if degenerate else centered @ axis
        order = np.lexsort((ids, scores))
        middle = len(ids)//2
        nodes.append({'level':level, 'prefix':leaf_prefix, 'records':len(ids),
                      'degenerate':bool(degenerate), 'axis_largest_coordinate':int(np.argmax(np.abs(axis))),
                      'left_max_projection':float(scores[order[middle-1]]),
                      'right_min_projection':float(scores[order[middle]])})
        split(ids[order[:middle]], level+1, leaf_prefix*2)
        split(ids[order[middle:]], level+1, leaf_prefix*2+1)

    split(np.arange(len(z), dtype=np.int64), 0, 0)
    return labels, {'algorithm':'balanced median PCA tree', 'depth':depth, 'groups':2**depth,
                    'power_iterations':24, 'seed':seed, 'uses_randomness':True,
                    'randomness_scope':'fixed per-node dense initialization; deterministic for fixed seed and inputs',
                    'tie_break':'parent row index', 'leaf_record_counts':leaf_sizes,
                    'seconds':time.monotonic()-started, 'nodes':nodes,
                    'scope':'target-data coverage strata, not validated independent capabilities'}


def nested_greedy(matrix, weights, q):
    """First q exact-lazy-greedy records form the reference; first 2q candidates."""
    started = time.monotonic()
    q = int(q)
    if q < 1 or 2*q > matrix.shape[1]:
        raise ValueError('Nested candidate budget must fit the leaf')
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != matrix.shape[0] or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError('Invalid demand weights')
    if np.any(matrix.data < 0) or not np.isfinite(matrix.data).all():
        raise ValueError('Coverage requires finite nonnegative similarities')
    order, _, trace, pops = lazy_greedy(matrix, weights, np.zeros(matrix.shape[1], np.int32),
                                      np.asarray([2*q], np.int64), progress=False)
    gains = trace[:,2]
    if np.any(gains < -1e-12) or np.any(np.diff(trace[:,3]) < -1e-9):
        raise AssertionError('Residual coverage declined')
    return order[:q].copy(), order.copy(), {
        'reference_objective':float(trace[q-1,3]/weights.sum()),
        'candidate_objective':float(trace[-1,3]/weights.sum()),
        'heap_pops':int(pops), 'greedy_seconds':time.monotonic()-started,
        'trace':trace.tolist(), 'normalization':'each record total demand weight one'}

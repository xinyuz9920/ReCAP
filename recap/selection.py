"""Portable CPU orchestration of the preserved selection core."""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from .backend import search
from .frontend import balanced_pca_groups, nested_greedy, normalized, record_features
from .graph import allocate, record_graph


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_features(path):
    """Reject malformed features before constructing graphs; no pickle loading."""
    with np.load(path, allow_pickle=False) as bundle:
        missing = set(['image', 'text', 'owner', 'direction'])-set(bundle.files)
        if missing:
            raise ValueError(f'Missing NPZ arrays: {sorted(missing)}')
        image, text = np.asarray(bundle['image']), np.asarray(bundle['text'])
        owner, direction = np.asarray(bundle['owner']), np.asarray(bundle['direction'])
    if image.ndim != 2 or text.ndim != 2 or len(image) == 0 or len(text) == 0:
        raise ValueError('image and text must be nonempty two-dimensional arrays')
    if image.shape[1] == 0 or text.shape[1] == 0:
        raise ValueError('Feature dimensions must be positive')
    for vectors in [image, text]:
        if vectors.dtype.kind not in 'fiu':
            raise ValueError('Features must be numeric')
        normalized(vectors)  # finite, nonzero rows; core normalizes them again
    for name, values in [('owner', owner), ('direction', direction)]:
        if values.ndim != 1 or len(values) != len(text) or values.dtype.kind not in 'iu':
            raise ValueError(f'{name} must be a one-dimensional integer array of length M')
        if np.any(values < 0):
            raise ValueError(f'{name} must be nonnegative')
    if np.any(owner >= len(image)):
        raise ValueError('owner is outside parent record indices')
    owner = owner.astype(np.int64)
    turns = np.bincount(owner, minlength=len(image))
    if np.any(turns == 0):
        raise ValueError('Every parent record needs at least one visible QA demand')
    return image.astype(np.float32), text.astype(np.float32), owner, direction.astype(np.int64), turns


def independent_coverage(matrix, selected):
    """CSR row max, without the backend incremental winner/runner-up state."""
    active = np.zeros(matrix.shape[1], dtype=bool)
    active[np.asarray(selected, dtype=np.int64)] = True
    values = np.where(active[matrix.indices], matrix.data, 0.0)
    covered = np.zeros(matrix.shape[0], dtype=np.float64)
    for row in range(matrix.shape[0]):
        left, right = matrix.indptr[row:row+2]
        if right > left:
            covered[row] = values[left:right].max()
    return covered


def validate_run(run):
    """Check integrity, nested budgets and the recomputed global-graph floors."""
    run = Path(run)
    manifest = read_json(run/'manifest.json')
    for name, digest in manifest['artifacts'].items():
        target = run/name
        if target.resolve().parent != run.resolve():
            raise ValueError('Unexpected artifact path')
        if sha(target) != digest:
            raise ValueError(f'Artifact changed: {name}')
    if manifest['tau'] != 1.0:
        raise ValueError('This release validates the tau=1 setting')
    candidate, reference, selected, groups, quotas, owner, weights = [
        np.load(run/f'{name}.npy', allow_pickle=False) for name in
        ['candidate', 'reference', 'selected', 'groups', 'quotas', 'owner', 'weights']]
    count, budget = manifest['records'], manifest['budget']
    for name, values, expected in [('candidate', candidate, 2*budget),
                                   ('reference', reference, budget), ('selected', selected, budget)]:
        if len(values) != expected or np.any(np.diff(values) <= 0) or values.min() < 0 or values.max() >= count:
            raise ValueError(f'Invalid {name} indices')
    if not np.isin(reference, candidate).all() or not np.isin(selected, candidate).all():
        raise ValueError('Reference/final membership must be nested inside candidates')
    np.testing.assert_array_equal(np.bincount(groups[reference], minlength=len(quotas)), quotas)
    np.testing.assert_array_equal(np.bincount(groups[candidate], minlength=len(quotas)), 2*quotas)
    np.testing.assert_allclose(np.bincount(owner, weights=weights, minlength=count), np.ones(count), atol=1e-12)
    frontend = read_json(run/'frontend.json')
    for leaf in frontend['leaves']:
        order, q = np.asarray(leaf['parent_sequence']), leaf['quota']
        np.testing.assert_array_equal(np.sort(order[:q]), reference[groups[reference] == leaf['group']])
        np.testing.assert_array_equal(np.sort(order), candidate[groups[candidate] == leaf['group']])
    matrix = sparse.load_npz(run/'graph.npz')
    if matrix.shape != (len(owner), len(candidate)):
        raise ValueError('Global graph shape mismatch')
    baseline = independent_coverage(matrix, np.searchsorted(candidate, reference))
    actual = independent_coverage(matrix, np.searchsorted(candidate, selected))
    demand_groups = groups[owner]
    mass = np.bincount(demand_groups, weights=weights, minlength=len(quotas))
    base_total = np.bincount(demand_groups, weights=weights*baseline, minlength=len(quotas))
    total = np.bincount(demand_groups, weights=weights*actual, minlength=len(quotas))
    tolerance = np.maximum(mass, 1)*1e-10
    if np.any(total < base_total-tolerance) or total.sum()+1e-8 < base_total.sum():
        raise ValueError('Global coverage or tau=1 group protection violated')
    report = read_json(run/'search.json')
    np.testing.assert_allclose(report['baseline_group_raw'], base_total, rtol=0, atol=1e-8)
    np.testing.assert_allclose(report['selected_group_raw'], total, rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.load(run/'coverage.npy'), actual, rtol=0, atol=1e-8)
    return {'status': 'passed', 'independent_csr_recomputation': True,
            'records': count, 'selected_records': budget, 'candidate_records': len(candidate),
            'groups': len(quotas), 'all_group_floors_pass': True,
            'reference_objective': float(base_total.sum()/count), 'objective': float(total.sum()/count),
            'minimum_positive_baseline_group_ratio': float(np.min(total[base_total > 0]/base_total[base_total > 0])) if np.any(base_total > 0) else None,
            'zero_baseline_groups': np.flatnonzero(base_total == 0).tolist(),
            'reference_intersection': int(np.isin(selected, reference).sum())}


def select(features, output, budget, depth=7, seed=42):
    """Select B distinct records using the original CPU kernel and frozen tau=1."""
    features, run = Path(features), Path(output)
    if run.exists() and any(run.iterdir()):
        raise ValueError('Output directory must be empty; use validate for an existing run')
    image, text, owner, direction, turns = load_features(features)
    budget, depth = int(budget), int(depth)
    if budget < 1 or depth < 0 or 2*budget > len(image) or 2**depth > len(image):
        raise ValueError('Require 1 <= 2*budget <= N and 0 <= depth with 2**depth <= N')
    started = time.monotonic()
    groups, partition = balanced_pca_groups(record_features(image, text, owner, turns), depth, seed)
    quotas = allocate(np.bincount(groups), budget)
    if np.any(quotas < 1) or np.any(2*quotas > np.bincount(groups)):
        raise ValueError('Each leaf requires 1 <= q and 2q <= leaf size; reduce depth or budget')
    weights = 1/turns[owner].astype(np.float64)
    run.mkdir(parents=True, exist_ok=True)
    write_json(run/'partition.json', partition)
    references, candidates, leaves = [], [], []
    front_start = time.monotonic()
    for group, q in enumerate(quotas):
        members = np.flatnonzero(groups == group)
        demands = np.flatnonzero(groups[owner] == group)
        local = np.full(len(image), -1, dtype=np.int64)
        local[members] = np.arange(len(members))
        matrix = record_graph(image[members], text[demands], local[owner[demands]], direction[demands],
                              np.arange(len(members)), k=32, device='cpu', progress=False)
        reference, candidate, report = nested_greedy(matrix, weights[demands], int(q))
        references.extend(members[reference].tolist())
        candidates.extend(members[candidate].tolist())
        leaves.append({'group': group, 'quota': int(q), 'parent_sequence': members[candidate].tolist(), **report})
    reference, candidate = np.sort(references).astype(np.int64), np.sort(candidates).astype(np.int64)
    write_json(run/'frontend.json', {'leaves': leaves, 'seconds': time.monotonic()-front_start})
    graph_start = time.monotonic()
    matrix = record_graph(image, text, owner, direction, candidate, k=32, device='cpu', progress=False)
    graph_seconds = time.monotonic()-graph_start
    chosen, coverage, stats = search(matrix, weights, groups[owner], np.searchsorted(candidate, reference),
                                     tau=1.0, group_count=len(quotas), progress=False)
    selected = np.sort(candidate[chosen])
    sparse.save_npz(run/'graph.npz', matrix)
    for name, values in [('groups', groups), ('quotas', quotas), ('candidate', candidate),
                         ('reference', reference), ('selected', selected), ('owner', owner),
                         ('weights', weights), ('coverage', coverage)]:
        np.save(run/f'{name}.npy', values)
    write_json(run/'search.json', stats)
    artifacts = {p.name: sha(p) for p in sorted(run.iterdir()) if p.is_file()}
    write_json(run/'manifest.json', {'format': 'recap-portable-cpu-v1', 'records': len(image),
               'demands': len(owner), 'budget': budget, 'depth': depth, 'seed': seed,
               'candidate_multiplier': 2, 'neighbors': 32, 'tau': 1.0, 'device': 'cpu',
               'input_sha256': sha(features), 'artifacts': artifacts,
               'graph_seconds': graph_seconds, 'core_seconds': time.monotonic()-started,
               'timing_excludes': 'feature extraction, input loading, export, training, evaluation',
               'implementation_sha256': {name: sha(Path(__file__).with_name(name))
                    for name in ['frontend.py', 'graph.py', 'greedy.py', 'backend.py', 'selection.py']}})
    result = validate_run(run)
    write_json(run/'validation.json', result)
    return result


def export_records(pool_path, run, output, pad_to=128, seed=42):
    """Export unchanged source records with the recorded shuffle/padding rule."""
    validation = validate_run(run)
    records = read_json(pool_path)
    if not isinstance(records, list) or len(records) != validation['records']:
        raise ValueError('Pool must be a JSON list in the original feature record order')
    if pad_to < 1:
        raise ValueError('pad_to must be positive')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Export output directory must be empty')
    indices = np.load(Path(run)/'selected.npy', allow_pickle=False).tolist()
    random.Random(seed).shuffle(indices)
    subset = [records[i] for i in indices]
    padding = (-len(subset)) % pad_to
    train = subset + [subset[i % len(subset)] for i in range(padding)]
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/'training_order.npy', np.asarray(indices, dtype=np.int64))
    write_json(output/'subset.json', subset)
    write_json(output/'train.json', train)
    result = {'unique_records': len(subset), 'padding_records': padding, 'training_records': len(train),
              'shuffle_seed': seed, 'pad_to': pad_to, 'pool_sha256': sha(pool_path),
              'selection_manifest_sha256': sha(Path(run)/'manifest.json'), 'train_sha256': sha(output/'train.json'),
              'mapping_contract': 'Caller must ensure pool row order matches feature image rows; row semantics cannot be inferred from vectors.'}
    write_json(output/'export.json', result)
    return result

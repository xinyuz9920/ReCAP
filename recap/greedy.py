"""Generic deterministic lazy facility-location greedy from OnceCover."""
import heapq
import time
import numpy as np

def lazy_greedy(matrix, weights, candidate_groups, quotas, progress=False):
    matrix = matrix.tocsc()
    weights = np.asarray(weights, np.float64)
    covered = np.zeros(matrix.shape[0], np.float64)
    used = np.zeros(len(quotas), np.int64)
    initial = np.asarray(matrix.T @ weights).ravel()
    heap = [(-float(g), j) for j, g in enumerate(initial)]
    heapq.heapify(heap)
    selected, trace = [], []
    pops, start = 0, time.monotonic()
    target = int(np.sum(quotas))
    while len(selected) < target:
        if not heap:
            raise ValueError("Candidate quotas cannot be filled")
        bound, j = heapq.heappop(heap)
        pops += 1
        group = candidate_groups[j]
        if used[group] >= quotas[group]:
            continue
        left, right = matrix.indptr[j:j+2]
        rows = matrix.indices[left:right]
        values = matrix.data[left:right]
        gain = float(weights[rows] @ np.maximum(values-covered[rows], 0))
        while heap and used[candidate_groups[heap[0][1]]] >= quotas[candidate_groups[heap[0][1]]]:
            heapq.heappop(heap)
            pops += 1
        # A stale upper bound on another candidate prevents premature acceptance.
        # Resolve numerical ties by original (sorted) candidate index.
        accept = not heap or (-gain, j) <= heap[0]
        if accept:
            selected.append(j)
            used[group] += 1
            covered[rows] = np.maximum(covered[rows], values)
            trace.append((len(selected), j, gain, float(weights @ covered)))
            if progress and (len(selected) % 512 == 0 or len(selected) == target):
                print(f"greedy {len(selected)}/{target} objective={trace[-1][3]:.6f} pops={pops} elapsed={time.monotonic()-start:.1f}s", flush=True)
        else:
            heapq.heappush(heap, (-gain, j))
    if not np.array_equal(used, quotas):
        raise AssertionError("Final quotas differ")
    return np.asarray(selected, np.int64), covered, np.asarray(trace), pops



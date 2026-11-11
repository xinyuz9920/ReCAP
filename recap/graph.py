"""Unchanged record-max before top-k coverage kernel from OnceCover."""
import time
import numpy as np
from scipy import sparse
import torch
import torch.nn.functional as F

def allocate(counts, budget):
    counts = np.asarray(counts, np.int64)
    if not 0 < budget <= counts.sum():
        raise ValueError("Budget outside capacity")
    exact = counts.astype(float) * budget / counts.sum()
    result = np.floor(exact).astype(np.int64)
    for j in np.lexsort((np.arange(len(counts)), -(exact-result)))[:budget-int(result.sum())]:
        result[j] += 1
    if result.sum() != budget or np.any(result > counts):
        raise AssertionError("Invalid Hamilton allocation")
    return result


def record_graph(image, text, owner, directions, candidate, k=32, batch_size=128, device="cpu", progress=True):
    """Rows=demands, columns=distinct candidate records. Top-k after max over turns."""
    image = F.normalize(torch.as_tensor(image, dtype=torch.float32, device=device), dim=1)
    text = F.normalize(torch.as_tensor(text, dtype=torch.float32, device=device), dim=1)
    owner = np.asarray(owner)
    directions = np.asarray(directions)
    candidate = np.asarray(candidate)
    local = np.full(len(image), -1, np.int32)
    local[candidate] = np.arange(len(candidate))
    k = min(int(k), len(candidate))
    cols = np.empty((len(owner), k), np.int32)
    values = np.zeros((len(owner), k), np.float32)
    candidate_image = image[torch.as_tensor(candidate, device=device)]
    start = time.monotonic()
    with torch.inference_mode():
        for direction in np.unique(directions):
            queries = np.flatnonzero(directions == direction)
            facilities = np.flatnonzero((directions == direction) & (local[owner] >= 0))
            if not len(facilities):
                raise ValueError(f"Direction {direction} has demand but no candidate; widen candidates")
            facility_owner = torch.as_tensor(local[owner[facilities]], dtype=torch.long, device=device)
            facility_text = text[torch.as_tensor(facilities, device=device)]
            for left in range(0, len(queries), batch_size):
                ids = queries[left:left+batch_size]
                query_ids = torch.as_tensor(ids, device=device)
                visual = image[torch.as_tensor(owner[ids], device=device)] @ candidate_image.T
                visual.clamp_(0, 1)
                sim = text[query_ids] @ facility_text.T
                sim.clamp_(0, 1)
                sim.mul_(visual[:, facility_owner])
                aggregate = torch.zeros((len(ids), len(candidate)), dtype=torch.float32, device=device)
                aggregate.scatter_reduce_(1, facility_owner[None, :].expand(len(ids), -1), sim, reduce="amax", include_self=True)
                take = min(k+1, len(candidate))
                best, indices = aggregate.topk(take, dim=1, sorted=True)
                # Resolve *membership* ties at positive k-boundary by original candidate row order.
                if take > k:
                    tied = (best[:, k-1] == best[:, k]) & (best[:, k-1] > 0)
                    for row in tied.nonzero().flatten().tolist():
                        stable = torch.argsort(aggregate[row], descending=True, stable=True)[:k]
                        indices[row, :k] = stable
                        best[row, :k] = aggregate[row, stable]
                cols[ids] = indices[:, :k].cpu().numpy()
                values[ids] = best[:, :k].cpu().numpy()
                if progress and (left // batch_size) % 64 == 0:
                    print(f"graph direction={direction} queries={min(left+batch_size,len(queries))}/{len(queries)} elapsed={time.monotonic()-start:.1f}s", flush=True)
                del visual, sim, aggregate, best, indices
    positive = values.ravel() > 0
    counts = (values > 0).sum(1)
    indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    matrix = sparse.csr_matrix((values.ravel()[positive], cols.ravel()[positive], indptr), shape=(len(owner), len(candidate)))
    matrix.sort_indices()
    return matrix



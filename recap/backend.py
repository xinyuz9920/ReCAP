"""Unchanged reference-protected bounded exchange solver from OnceCover."""
import itertools
import time
import numpy as np
from scipy import sparse

SEARCH = {"add_width": 16, "remove_width": 64, "fallback_pairs": 48,
          "joint_add_width": 6, "joint_remove_width": 8, "joint_trials": 48,
          "max_exchanges": 16384, "minimum_raw_gain": 1e-7}


def sparse_state(matrix, selected, rows=None):
    """Best value, lowest-index winning record, runner-up excluding winner."""
    rows = np.arange(matrix.shape[0]) if rows is None else np.asarray(rows)
    best = np.zeros(len(rows), np.float64)
    second = np.zeros(len(rows), np.float64)
    winner = np.full(len(rows), matrix.shape[1], np.int32)
    for left in range(0, len(rows), 20000):
        sub = rows[left:left+20000]
        lengths = np.diff(matrix.indptr)[sub]
        width = int(lengths.max(initial=0))
        if not width:
            continue
        offsets = np.arange(width)[None, :]
        valid = offsets < lengths[:, None]
        loc = np.minimum(matrix.indptr[sub, None]+offsets, max(matrix.nnz-1, 0))
        cols = matrix.indices[loc]
        values = np.where(valid & selected[cols], matrix.data[loc], 0.0)
        top = values.max(1)
        win = np.where(valid & selected[cols] & (values == top[:, None]) & (top[:, None] > 0),
                       cols, matrix.shape[1]).min(1)
        runner = np.where(cols != win[:, None], values, 0).max(1)
        best[left:left+len(sub)] = top
        winner[left:left+len(sub)] = win
        second[left:left+len(sub)] = runner
    return best, winner, second


class ExchangeState:
    def __init__(self, matrix, weights, groups, initial, group_count=None):
        self.matrix = matrix.tocsr(copy=True)
        self.matrix.sort_indices()
        self.csc = self.matrix.tocsc()
        self.weights = np.asarray(weights, np.float64)
        self.groups = np.asarray(groups, np.int32)
        self.G = int(group_count or (self.groups.max()+1))
        self.C = matrix.shape[1]
        self.selected = np.zeros(self.C, bool)
        self.selected[np.asarray(initial, np.int64)] = True
        self.edge_rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int32), np.diff(self.matrix.indptr))
        self.best, self.winner, self.second = sparse_state(self.matrix, self.selected)
        self.group_total = np.bincount(self.groups, weights=self.weights*self.best, minlength=self.G)
        self.group_gain = np.zeros((self.C+1, self.G), np.float64)
        self.group_loss = np.zeros((self.C+1, self.G), np.float64)
        self.gain = np.zeros(self.C+1, np.float64)
        self.loss = np.zeros(self.C+1, np.float64)
        er = self.edge_rows
        delta = self.weights[er]*np.maximum(self.matrix.data-self.best[er], 0)
        self.gain[:self.C] = np.bincount(self.matrix.indices, weights=delta, minlength=self.C)
        flat = self.matrix.indices.astype(np.int64)*self.G+self.groups[er]
        self.group_gain[:self.C] = np.bincount(flat, weights=delta, minlength=self.C*self.G).reshape(self.C,self.G)
        drops = self.weights*(self.best-self.second)
        np.add.at(self.loss,self.winner,drops)
        np.add.at(self.group_loss,(self.winner,self.groups),drops)

    def affected(self, outgoing, incoming):
        blocks = [self.csc.indices[self.csc.indptr[c]:self.csc.indptr[c+1]] for c in list(outgoing)+list(incoming)]
        return np.unique(np.concatenate(blocks)) if blocks else np.empty(0,np.int32)

    def trial(self, outgoing, incoming, floor, tolerance, minimum_gain=1e-7):
        outgoing, incoming = tuple(outgoing), tuple(incoming)
        if len(outgoing)!=len(incoming) or not outgoing:
            raise ValueError('Exchange must preserve nonempty cardinality')
        if not self.selected[list(outgoing)].all() or self.selected[list(incoming)].any():
            raise ValueError('Invalid exchange membership')
        rows = self.affected(outgoing,incoming)
        self.selected[list(outgoing)] = False
        self.selected[list(incoming)] = True
        try:
            best,winner,second = sparse_state(self.matrix,self.selected,rows)
        finally:
            self.selected[list(incoming)] = False
            self.selected[list(outgoing)] = True
        delta = self.weights[rows]*(best-self.best[rows])
        group_delta = np.bincount(self.groups[rows],weights=delta,minlength=self.G)
        gain = float(delta.sum())
        if gain <= minimum_gain or np.any(self.group_total+group_delta < floor-tolerance):
            return None
        return (outgoing,incoming,rows,best,winner,second,group_delta,gain)

    def accept(self, proposal):
        outgoing,incoming,rows,best,winner,second,group_delta,gain = proposal
        old_best = self.best[rows].copy()
        old_drop = self.weights[rows]*(old_best-self.second[rows])
        np.add.at(self.loss,self.winner[rows],-old_drop)
        np.add.at(self.group_loss,(self.winner[rows],self.groups[rows]),-old_drop)
        # Incrementally update all addition gains touched by changed demand rows.
        lengths = np.diff(self.matrix.indptr)[rows]
        width = int(lengths.max(initial=0))
        offsets = np.arange(width)[None,:]
        valid = offsets < lengths[:,None]
        ids = (self.matrix.indptr[rows,None]+offsets)[valid]
        rr = np.repeat(rows,lengths)
        old_rows = np.repeat(old_best,lengths)
        new_rows = np.repeat(best,lengths)
        vals = self.matrix.data[ids]
        diff = self.weights[rr]*(np.maximum(vals-new_rows,0)-np.maximum(vals-old_rows,0))
        cols = self.matrix.indices[ids]
        np.add.at(self.gain,cols,diff)
        np.add.at(self.group_gain,(cols,self.groups[rr]),diff)
        self.selected[list(outgoing)] = False
        self.selected[list(incoming)] = True
        self.best[rows],self.winner[rows],self.second[rows] = best,winner,second
        new_drop = self.weights[rows]*(best-second)
        np.add.at(self.loss,winner,new_drop)
        np.add.at(self.group_loss,(winner,self.groups[rows]),new_drop)
        self.group_total += group_delta
        return gain


def search(matrix,weights,groups,initial,tau,group_count=None,settings=None,progress=False):
    cfg = SEARCH | (settings or {})
    state = ExchangeState(matrix,weights,groups,initial,group_count)
    baseline = state.group_total.copy()
    floor = float(tau)*baseline
    # Numerical tolerance corresponds to at most 1e-10 in per-record group coverage.
    group_mass = np.bincount(groups,weights=weights,minlength=state.G)
    tolerance = np.maximum(group_mass,1)*1e-10
    trace=[]; trials=0; joint_count=0; started=time.monotonic();stop='exchange_budget'
    for iteration in range(int(cfg['max_exchanges'])):
        add = np.flatnonzero(~state.selected)
        remove = np.flatnonzero(state.selected)
        add = add[np.lexsort((add,-state.gain[add]))[:cfg['add_width']]]
        remove = remove[np.lexsort((remove,state.loss[remove]))[:cfg['remove_width']]]
        if not len(add) or not len(remove): stop='no_membership_choices';break
        slack = state.group_total-floor
        lower_groups = slack[None,None,:]+state.group_gain[add,None,:]-state.group_loss[None,remove,:]
        lower_gain = state.gain[add,None]-state.loss[None,remove]
        feasible = (lower_groups >= -tolerance).all(2) & (lower_gain > cfg['minimum_raw_gain'])
        feasible_pairs = np.argwhere(feasible)
        proposal=None
        if len(feasible_pairs):
            scores=lower_gain[feasible]
            order=np.lexsort((remove[feasible_pairs[:,1]],add[feasible_pairs[:,0]],-scores))
            for ai,oi in feasible_pairs[order[:8]]:
                trials+=1
                proposal=state.trial([int(remove[oi])],[int(add[ai])],floor,tolerance,cfg['minimum_raw_gain'])
                if proposal is not None:break
        if proposal is None:
            # Exact fallback checks also consider overlap corrections omitted by lower bounds.
            deficit=np.maximum(-lower_groups,0)/np.maximum(group_mass,1)[None,None,:]
            deficit=deficit.sum(2)
            pairs=[(float(deficit[a,o]),-float(lower_gain[a,o]),int(add[a]),int(remove[o]))
                   for a in range(len(add)) for o in range(len(remove))]
            for _,__,a,o in sorted(pairs)[:cfg['fallback_pairs']]:
                trials+=1
                proposal=state.trial([o],[a],floor,tolerance,cfg['minimum_raw_gain'])
                if proposal is not None:break
        if proposal is None:
            # Bounded two-for-two lookahead; exact group feasibility is checked before acceptance.
            a_pairs=list(itertools.combinations(add[:cfg['joint_add_width']].tolist(),2))
            o_pairs=list(itertools.combinations(remove[:cfg['joint_remove_width']].tolist(),2))
            candidates=[]
            for aa in a_pairs:
                ag=state.group_gain[list(aa)].sum(0)
                for oo in o_pairs:
                    approximate=slack+ag-state.group_loss[list(oo)].sum(0)
                    deficit=float((np.maximum(-approximate,0)/np.maximum(group_mass,1)).sum())
                    score=float(state.gain[list(aa)].sum()-state.loss[list(oo)].sum())
                    candidates.append((deficit,-score,aa,oo))
            for _,__,aa,oo in sorted(candidates)[:cfg['joint_trials']]:
                trials+=1
                proposal=state.trial(oo,aa,floor,tolerance,cfg['minimum_raw_gain'])
                if proposal is not None:joint_count+=1;break
        if proposal is None:stop='bounded_neighborhood_exhausted';break
        gain=state.accept(proposal)
        trace.append({'iteration':iteration+1,'out':list(proposal[0]),'in':list(proposal[1]),
                      'raw_gain':gain,'raw_objective':float(state.group_total.sum()),
                      'min_group_ratio':float(np.min(state.group_total/np.maximum(baseline,1e-30)))})
        if progress and (len(trace)%128==0 or len(trace)==1):
            print(f'tau={tau} exchanges={len(trace)} raw_objective={state.group_total.sum():.6f} min_ratio={trace[-1]["min_group_ratio"]:.6f} elapsed={time.monotonic()-started:.1f}s',flush=True)
    # Independently recompute from the final membership; incremental state is not the verifier.
    actual,_,_=sparse_state(matrix,state.selected)
    actual_groups=np.bincount(groups,weights=np.asarray(weights)*actual,minlength=state.G)
    if np.max(np.abs(actual-state.best))>1e-10 or np.any(actual_groups < floor-tolerance):
        raise AssertionError('Final independently recomputed coverage violates state or floor')
    if actual_groups.sum()+1e-8 < baseline.sum():
        raise AssertionError('Coverage declined from reference')
    return np.flatnonzero(state.selected),actual,{'tau':float(tau),'settings':cfg,'stop_reason':stop,
        'accepted_exchanges':len(trace),'two_for_two_exchanges':joint_count,'exact_trials':trials,
        'seconds':time.monotonic()-started,'baseline_group_raw':baseline.tolist(),
        'selected_group_raw':actual_groups.tolist(),'group_mass':group_mass.tolist(),
        'minimum_group_ratio':float(np.min(actual_groups/np.maximum(baseline,1e-30))),
        'raw_objective':float(actual_groups.sum()),'baseline_raw_objective':float(baseline.sum()),'trace':trace,
        'constraint_tolerance_per_record':1e-10,'optimization_scope':'bounded deterministic 1-for-1 and 2-for-2 local search; no global optimality claim'}



# ReCAP selection semantics

Let record `i` own `m_i` training-visible QA demands. Its demand weights are `w_ir = 1/m_i`, so each record has total weight one. The partition representation concatenates its normalized image vector and the normalized mean of its normalized visible-QA text vectors, followed by normalization. A balanced median PCA tree with fixed per-node seeded initialization yields `2^depth` strata. Hamilton allocation gives quotas `q_g` summing to budget `B`.

For a demand `a=(i,r)` and candidate record `j`, use

```math
k_{a,j}=\max_{s\in\mathcal A_j}\mathbf1[d_a=d_{js}]
       [v_i^\top v_j]_+[t_a^\top t_{js}]_+.
```

Maximization inside each record precedes top-32 record truncation. A long conversation can contribute multiple demands but cannot occupy multiple facility slots in the same demand's neighbor list.

Each stratum builds its own local graph and performs exact lazy cardinality greedy for `2*q_g` records. Its first `q_g` records form the reference, and its full prefix forms the candidates. Union over strata gives `|S0|=B` and `|C|=2B`.

Rebuild the global graph from **all parent demands** to the union of candidates. Do not filter or reuse only the local edges. Let the fixed sparse kernel be `k_tilde`. The coverage proxy and raw group coverage are

```math
F(S)=\frac1N\sum_i\frac1{m_i}\sum_{r\in\mathcal A_i}
                 \max_{j\in S}\widetilde{k}_{ir,j},
\qquad H_g(S)=\sum_{i\in g}\frac1{m_i}\sum_r\max_{j\in S}\widetilde{k}_{ir,j}.
```

Recompute `H_g(S0)` on this new graph. Starting at `S0`, maximize `F(S)` at fixed cardinality `B`, subject to `H_g(S) >= H_g(S0)` for every stratum (`tau=1`). The preserved solver ranks bounded incoming/outgoing candidates, tests exact one-for-one exchanges and then bounded two-for-two exchanges, accepts strict objective improvements, and independently recomputes final coverage. It neither fixes final per-stratum record counts nor certifies a global optimum.

The implementation retains the original numerical tolerances. Independent public tests check dense greedy equivalence, record-max/top-k ordering, direction and image gating, stable boundary ties, nonpositive exchange rejection, protected group floors, incremental state reconstruction, and a joint exchange that succeeds when every single exchange is infeasible.

## Representation contract and limits

The original feature preparation used frozen CLIP ViT-B/32 whole-image and text embeddings with a 512-dimensional representation. Ordinary QA text used the current question with at most the previous QA context, without the current answer; region-to-description demands used the visible reference description, and description-to-coordinate demands used description text without target coordinates. This portable selector intentionally does not invent a generic encoder or visibility parser that would silently change those semantics. Supply equivalent features for the original experiment, or explicitly report a new representation protocol.

No target-VLM forward pass, gradient or likelihood score is used by the selector. The residual means uncovered semantic demand, not model uncertainty. Whole-image features can confuse different regions or opposite short answers. PCA strata are semantic partitions rather than proven independent abilities. Coverage floors protect only this fixed proxy graph.

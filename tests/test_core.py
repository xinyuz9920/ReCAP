"""Independent small-instance checks for ResidualCover's selection semantics.

These are CPU-only checks. Dense exhaustive calculations are deliberately used
as an oracle; no expected outputs are copied from production reports.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from recap.graph import allocate, record_graph
from recap.greedy import lazy_greedy
from recap.backend import ExchangeState, search
from recap.frontend import balanced_pca_groups, record_features, nested_greedy


def dense_greedy(matrix, weights, count):
    """Exhaustive cardinality-greedy oracle, with lowest column as tie breaker."""
    matrix = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    covered = np.zeros(matrix.shape[0], dtype=np.float64)
    order = []
    for _ in range(count):
        gains = weights @ np.maximum(matrix - covered[:, None], 0)
        gains[np.asarray(order, dtype=int)] = -np.inf
        chosen = int(np.argmax(gains))
        order.append(chosen)
        covered = np.maximum(covered, matrix[:, chosen])
    return np.asarray(order), covered


class TargetPartitionTests(unittest.TestCase):
    def test_balanced_tree_is_a_deterministic_complete_partition(self):
        rng = np.random.default_rng(543)
        features = rng.normal(size=(257, 11)).astype(np.float32)
        groups, report = balanced_pca_groups(features, depth=4, seed=42)
        repeat, _ = balanced_pca_groups(features, depth=4, seed=42)
        groups = np.asarray(groups)
        np.testing.assert_array_equal(groups, repeat)
        np.testing.assert_array_equal(np.unique(groups), np.arange(16))
        counts = np.bincount(groups, minlength=16)
        self.assertEqual(int(counts.sum()), 257)
        self.assertLessEqual(int(counts.max()-counts.min()), 1)

    def test_degenerate_features_have_reproducible_balanced_tie_breaks(self):
        features = np.zeros((64, 7), np.float32)
        groups, _ = balanced_pca_groups(features, depth=4, seed=42)
        repeat, _ = balanced_pca_groups(features.copy(), depth=4, seed=42)
        np.testing.assert_array_equal(groups, repeat)
        np.testing.assert_array_equal(np.bincount(np.asarray(groups)), np.full(16, 4))

    def test_record_representation_does_not_depend_on_qa_storage_order(self):
        rng = np.random.default_rng(721)
        image = rng.normal(size=(3, 5)).astype(np.float32)
        text = rng.normal(size=(6, 5)).astype(np.float32)
        owner = np.array([0, 1, 1, 2, 2, 2])
        actual = np.asarray(record_features(image, text, owner))
        permutation = np.array([5, 2, 0, 4, 1, 3])
        repeat = np.asarray(record_features(image, text[permutation], owner[permutation]))
        self.assertEqual(actual.shape[0], 3)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, repeat, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), np.ones(3), atol=1e-6)

    def test_nested_frontend_matches_dense_oracle_and_keeps_prefix(self):
        rng = np.random.default_rng(414)
        dense = rng.random((17, 12))
        dense[rng.random(dense.shape) < .5] = 0
        weights = rng.random(17) + .01
        reference, candidate, _ = nested_greedy(sparse.csr_matrix(dense), weights, 4)
        expected, _ = dense_greedy(dense, weights, 8)
        np.testing.assert_array_equal(candidate, expected)
        np.testing.assert_array_equal(reference, expected[:4])
        self.assertEqual(len(set(np.asarray(candidate).tolist())), 8)
        self.assertTrue(set(reference).issubset(set(candidate)))


class RecordKernelTests(unittest.TestCase):
    def test_record_max_precedes_topk(self):
        # Record 1 has the top two QA matches. Record 2 must nevertheless remain
        # a distinct second facility; top-k QA followed by de-dup would lose it.
        image = np.tile([1.0, 0.0], (3, 1))
        text = np.array([[1, 0], [1, 0], [.999, .04], [.6, .8]], np.float32)
        owner = np.array([0, 1, 1, 2])
        actual = record_graph(image, text, owner, np.zeros(4, int),
                              np.array([1, 2]), k=2, batch_size=2,
                              device="cpu", progress=False).toarray()
        self.assertEqual(actual.shape, (4, 2))
        np.testing.assert_allclose(actual[0], [1.0, .6], atol=2e-6)
        self.assertEqual(np.count_nonzero(actual[0]), 2)

    def test_direction_and_image_are_both_required(self):
        image = np.array([[1, 0], [1, 0], [-1, 0]], np.float32)
        text = np.tile([1., 0.], (3, 1))
        owner = np.arange(3)
        actual = record_graph(image, text, owner, np.array([0, 1, 0]),
                              np.arange(3), k=3, batch_size=2,
                              device="cpu", progress=False).toarray()
        np.testing.assert_allclose(actual[0], [1, 0, 0], atol=2e-6)

    def test_positive_boundary_ties_are_deterministic(self):
        image = np.tile([1., 0.], (5, 1))
        text = image.copy()
        owner = np.arange(5)
        kwargs = dict(k=2, batch_size=2, device="cpu", progress=False)
        actual = record_graph(image, text, owner, np.zeros(5, int),
                              np.array([1, 2, 3, 4]), **kwargs)
        np.testing.assert_array_equal(actual.indices.reshape(-1, 2),
                                      np.tile([0, 1], (5, 1)))
        repeat = record_graph(image, text, owner, np.zeros(5, int),
                              np.array([1, 2, 3, 4]), **kwargs)
        np.testing.assert_array_equal(actual.indptr, repeat.indptr)
        np.testing.assert_array_equal(actual.indices, repeat.indices)
        np.testing.assert_array_equal(actual.data, repeat.data)

    def test_missing_direction_fails_instead_of_silently_losing_demands(self):
        image = np.tile([1., 0.], (2, 1))
        with self.assertRaises(ValueError):
            record_graph(image, image, np.arange(2), np.array([0, 1]),
                         np.array([0]), device="cpu", progress=False)

    def test_self_is_valid_coverage_but_does_not_duplicate_a_facility(self):
        image = np.array([[1, 0], [0, 1]], np.float32)
        text = np.array([[1, 0], [0, 1], [.6, .8], [.8, .6]], np.float32)
        owner = np.array([0, 0, 1, 1])
        matrix = record_graph(image, text, owner, np.zeros(4, int), np.arange(2),
                              k=2, device="cpu", progress=False)
        np.testing.assert_allclose(matrix.toarray()[np.arange(4), owner], 1, atol=2e-6)
        self.assertTrue(np.all(np.diff(matrix.indptr) <= 2))


class BudgetAndGreedyTests(unittest.TestCase):
    def test_real_budget_is_exact_and_double_prefix_fits_every_leaf(self):
        # 62,461 records partitioned into 128 balanced leaves.
        counts = np.full(128, 62461 // 128, dtype=np.int64)
        counts[:62461 % 128] += 1
        quotas = allocate(counts, 9369)
        self.assertEqual(int(quotas.sum()), 9369)
        self.assertEqual(int((2 * quotas).sum()), 18738)
        self.assertTrue(np.all(2 * quotas <= counts))
        self.assertTrue(np.all(quotas > 0))

    def test_lazy_greedy_matches_independent_dense_oracle(self):
        rng = np.random.default_rng(2841)
        dense = rng.random((19, 13))
        dense[rng.random(dense.shape) < .65] = 0
        weights = rng.random(19) + .1
        expected, covered = dense_greedy(dense, weights, 8)
        actual, actual_covered, trace, _ = lazy_greedy(
            sparse.csr_matrix(dense), weights, np.zeros(13, int), np.array([8]))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(actual_covered, covered, atol=1e-12)
        self.assertTrue(np.all(np.diff(np.r_[0, trace[:, 3]]) >= -1e-12))
        for length, column, gain, objective in trace:
            length = int(length)
            selected = actual[:length]
            independent = float(weights @ dense[:, selected].max(axis=1))
            self.assertAlmostEqual(float(objective), independent, places=11)
            previous = 0.0 if length == 1 else float(trace[length-2, 3])
            self.assertAlmostEqual(float(gain), independent - previous, places=11)

    def test_nested_prefix_is_same_as_independent_short_run(self):
        rng = np.random.default_rng(67)
        dense = rng.random((17, 10))
        weights = rng.random(17) + .01
        long, _, _, _ = lazy_greedy(sparse.csr_matrix(dense), weights,
                                   np.zeros(10, int), np.array([8]))
        short, _, _, _ = lazy_greedy(sparse.csr_matrix(dense), weights,
                                    np.zeros(10, int), np.array([4]))
        np.testing.assert_array_equal(long[:4], short)
        self.assertEqual(len(set(long.tolist())), 8)
        self.assertTrue(set(short).issubset(set(long)))

    def test_each_record_has_total_demand_one(self):
        owner = np.array([0, 1, 1, 1])
        turns = np.bincount(owner)
        weights = 1.0 / turns[owner]
        np.testing.assert_allclose(np.bincount(owner, weights=weights), [1, 1])
        dense = np.array([[1, 0], [0, .9], [0, .9], [0, .9]])
        selected, _, _, _ = lazy_greedy(sparse.csr_matrix(dense), weights,
                                       np.zeros(2, int), np.array([1]))
        self.assertEqual(int(selected[0]), 0)
        unweighted, _, _, _ = lazy_greedy(sparse.csr_matrix(dense), np.ones(4),
                                         np.zeros(2, int), np.array([1]))
        self.assertEqual(int(unweighted[0]), 1)

    def test_flat_marginals_fill_budget_deterministically(self):
        # Flat marginals are permitted in the FRONTEND to fulfill exact budgets.
        # They must never be accepted as backend exchanges (tested below).
        matrix = sparse.csr_matrix(np.ones((3, 6)))
        selected, _, trace, _ = lazy_greedy(matrix, np.ones(3),
                                           np.zeros(6, int), np.array([4]))
        np.testing.assert_array_equal(selected, [0, 1, 2, 3])
        np.testing.assert_array_equal(trace[1:, 2], 0)


class FrozenExchangeTests(unittest.TestCase):
    def test_zero_and_negative_gain_exchanges_are_rejected(self):
        for dense in (np.array([[1., 1.], [.2, .2]]),
                      np.array([[1., .5], [.2, .1]])):
            state = ExchangeState(sparse.csr_matrix(dense), np.ones(2),
                                  np.zeros(2, int), [0], 1)
            before = state.selected.copy()
            self.assertIsNone(state.trial([0], [1], np.zeros(1),
                                           np.zeros(1), minimum_gain=0))
            np.testing.assert_array_equal(state.selected, before)

    def test_global_improvement_cannot_violate_one_group_floor(self):
        dense = np.array([[.9, .8], [.1, .8]])
        state = ExchangeState(sparse.csr_matrix(dense), np.ones(2),
                              np.array([0, 1]), [0], 2)
        self.assertIsNone(state.trial([0], [1], state.group_total.copy(),
                                       np.zeros(2)))
        self.assertIsNotNone(state.trial([0], [1], np.zeros(2), np.zeros(2)))

    def test_incremental_state_matches_fresh_independent_reconstruction(self):
        rng = np.random.default_rng(901)
        dense = rng.random((23, 11))
        dense[rng.random(dense.shape) < .35] = 0
        matrix = sparse.csr_matrix(dense)
        weights = rng.random(23) + .1
        groups = np.arange(23) % 4
        state = ExchangeState(matrix, weights, groups, [0, 1, 2], 4)
        accepted = 0
        for _ in range(10):
            proposal = None
            for outgoing in np.flatnonzero(state.selected):
                for incoming in np.flatnonzero(~state.selected):
                    proposal = state.trial([int(outgoing)], [int(incoming)],
                                           np.zeros(4), np.zeros(4))
                    if proposal is not None:
                        break
                if proposal is not None:
                    break
            if proposal is None:
                break
            state.accept(proposal)
            accepted += 1
            selected = np.flatnonzero(state.selected)
            independent_covered = dense[:, selected].max(axis=1)
            np.testing.assert_allclose(state.best, independent_covered, atol=1e-12)
            fresh = ExchangeState(matrix, weights, groups, selected, 4)
            for name in ("best", "second", "gain", "loss", "group_gain",
                         "group_loss", "group_total"):
                np.testing.assert_allclose(getattr(state, name),
                                           getattr(fresh, name), atol=1e-10)
            np.testing.assert_array_equal(state.winner, fresh.winner)
        self.assertGreater(accepted, 0)

    def test_backend_recomputes_reference_on_its_actual_graph(self):
        # Leaf/coarse graph baseline would be [1, 0], but the rebuilt full graph
        # has [0.4, 0.3]. Only the latter can legally define this search's floor.
        full = np.array([[.4, .8], [.3, .9]])
        chosen, covered, report = search(sparse.csr_matrix(full), np.ones(2),
                                         np.array([0, 1]), [0], 1.0, 2)
        np.testing.assert_allclose(report["baseline_group_raw"], [.4, .3])
        np.testing.assert_array_equal(chosen, [1])
        np.testing.assert_allclose(covered, [.8, .9])
        self.assertTrue(all(row["raw_gain"] > 1e-7 for row in report["trace"]))

    def test_final_floor_and_objective_match_dense_oracle(self):
        rng = np.random.default_rng(194)
        dense = rng.random((31, 12))
        dense[rng.random(dense.shape) < .4] = 0
        weights = rng.random(31) + .1
        groups = np.arange(31) % 5
        initial = [0, 1, 2, 3]
        chosen, covered, report = search(sparse.csr_matrix(dense), weights,
                                         groups, initial, 1.0, 5)
        self.assertEqual(len(chosen), len(initial))
        self.assertEqual(len(set(chosen)), len(initial))
        expected = dense[:, chosen].max(axis=1)
        np.testing.assert_allclose(covered, expected, atol=1e-12)
        baseline = np.bincount(groups, weights=weights*dense[:, initial].max(1),
                               minlength=5)
        actual = np.bincount(groups, weights=weights*expected, minlength=5)
        tolerance = np.maximum(np.bincount(groups, weights=weights, minlength=5), 1)*1e-10
        self.assertTrue(np.all(actual >= baseline-tolerance))
        self.assertGreaterEqual(float(actual.sum())+1e-8, float(baseline.sum()))
        np.testing.assert_allclose(report["selected_group_raw"], actual, atol=1e-10)

    def test_joint_swap_can_preserve_floors_when_every_single_swap_fails(self):
        dense = np.array([[.8, 0, .9, 0], [.8, 0, 0, .9],
                          [0, .8, .9, 0], [0, .8, 0, .9]])
        matrix = sparse.csr_matrix(dense)
        state = ExchangeState(matrix, np.ones(4), np.arange(4), [0, 1], 4)
        floor = state.group_total.copy()
        for outgoing in [0, 1]:
            for incoming in [2, 3]:
                self.assertIsNone(state.trial([outgoing], [incoming], floor, np.zeros(4)))
        joint = state.trial([0, 1], [2, 3], floor, np.zeros(4))
        self.assertIsNotNone(joint)
        chosen, covered, report = search(matrix, np.ones(4), np.arange(4), [0, 1], 1.0, 4)
        np.testing.assert_array_equal(chosen, [2, 3])
        np.testing.assert_allclose(covered, np.full(4, .9), atol=1e-12)
        self.assertEqual(report["two_for_two_exchanges"], 1)



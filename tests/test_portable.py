import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from recap.selection import export_records, load_features, select, validate_run


class PortableTests(unittest.TestCase):
    def features(self, root):
        rng = np.random.default_rng(981)
        owner = np.repeat(np.arange(32), 1+np.arange(32) % 3)
        path = root/'features.npz'
        np.savez(path, image=rng.normal(size=(32, 9)), text=rng.normal(size=(len(owner), 9)),
                 owner=owner, direction=np.zeros(len(owner), dtype=np.int64))
        return path

    def test_end_to_end_and_independent_dense_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = self.features(root)
            with contextlib.redirect_stdout(io.StringIO()):
                result = select(features, root/'run', 8, depth=2)
            run = root/'run'
            matrix = sparse.load_npz(run/'graph.npz').toarray()
            candidate = np.load(run/'candidate.npy')
            reference = np.load(run/'reference.npy')
            selected = np.load(run/'selected.npy')
            weights = np.load(run/'weights.npy')
            groups = np.load(run/'groups.npy')[np.load(run/'owner.npy')]
            baseline = matrix[:, np.searchsorted(candidate, reference)].max(axis=1)
            coverage = matrix[:, np.searchsorted(candidate, selected)].max(axis=1)
            base_groups = np.bincount(groups, weights=weights*baseline)
            final_groups = np.bincount(groups, weights=weights*coverage)
            self.assertTrue(np.all(final_groups >= base_groups-1e-9))
            np.testing.assert_allclose(np.load(run/'coverage.npy'), coverage, atol=1e-12)
            self.assertAlmostEqual(result['objective'], float(weights @ coverage/32), places=12)
            self.assertEqual(result, validate_run(run))
            # Same fixed input/seed/core must produce the same memberships.
            with contextlib.redirect_stdout(io.StringIO()):
                select(features, root/'repeat', 8, depth=2)
            for name in ['groups', 'candidate', 'reference', 'selected']:
                np.testing.assert_array_equal(np.load(run/f'{name}.npy'), np.load(root/'repeat'/f'{name}.npy'))
            pool = [{'id': i, 'conversations': [{'value': f'untouched-{i}'}]} for i in range(32)]
            (root/'pool.json').write_text(json.dumps(pool), encoding='utf-8')
            exported = export_records(root/'pool.json', run, root/'export')
            self.assertEqual(exported['training_records'], 128)
            order = np.load(root/'export/training_order.npy')
            subset = json.loads((root/'export/subset.json').read_text())
            training = json.loads((root/'export/train.json').read_text())
            self.assertEqual(subset, [pool[i] for i in order])
            self.assertEqual(training, subset+[subset[i % len(subset)] for i in range(120)])
            bad = np.load(run/'selected.npy').copy()
            bad[0] = bad[-1]
            np.save(run/'selected.npy', bad)
            with self.assertRaisesRegex(ValueError, 'Artifact changed'):
                validate_run(run)

    def test_input_schema_rejects_missing_owner_and_fractional_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'bad.npz'
            np.savez(path, image=np.ones((3, 2)), text=np.ones((3, 2)),
                     owner=np.array([0, 1, 1]), direction=np.zeros(3, int))
            with self.assertRaisesRegex(ValueError, 'Every parent record'):
                load_features(path)
            np.savez(path, image=np.ones((3, 2)), text=np.ones((3, 2)),
                     owner=np.array([0., 1., 2.]), direction=np.zeros(3, int))
            with self.assertRaisesRegex(ValueError, 'integer array'):
                load_features(path)
            np.savez(path, image=np.ones((3, 2)), text=np.zeros((3, 2)),
                     owner=np.arange(3), direction=np.zeros(3, int))
            with self.assertRaisesRegex(ValueError, 'Zero feature vector'):
                load_features(path)

    def test_invalid_leaf_capacity_and_existing_run_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = self.features(root)
            with self.assertRaisesRegex(ValueError, 'reduce depth'):
                select(features, root/'invalid', 8, depth=4)
            (root/'existing').mkdir()
            (root/'existing/file').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'must be empty'):
                select(features, root/'existing', 8, depth=2)


if __name__ == '__main__':
    unittest.main()

"""
test_refactor.py
----------------
Smoke tests for the HCP + MTR performance refactor.
All tests run on CPU with tiny tensors (< 5 s, no GPU required).
Imports of hcp_project modules are deferred inside each test method so that
collection succeeds even in environments where some optional packages
(networkx, shapely, etc.) are absent.

Run with:
    python -m pytest hcp_project/tests/test_refactor.py -v
or:
    python hcp_project/tests/test_refactor.py
"""

import unittest
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Vectorised polyline transform
# ---------------------------------------------------------------------------
class TestTransformPolylinesBatch(unittest.TestCase):
    """
    Verify transform_polylines_batch produces numerically identical
    results to the original per-point loop.
    """

    @staticmethod
    def _ref_transform(polylines_raw, ego_x, ego_y, ego_heading):
        """Original per-point loop."""
        import math
        cos_h = math.cos(ego_heading)
        sin_h = math.sin(ego_heading)
        result = []
        for poly in polylines_raw:
            pts = []
            for pt in poly:
                dx = pt[0] - ego_x
                dy = pt[1] - ego_y
                x_n =  dx * cos_h + dy * sin_h
                y_n = -dx * sin_h + dy * cos_h
                pts.append([x_n, y_n, pt[2]])
            result.append(np.array(pts, dtype=np.float32))
        return result

    def test_matches_reference(self):
        # Deferred import so collection never fails
        from hcp_project.data.dataset_router import transform_polylines_batch

        rng = np.random.default_rng(0)
        polylines_raw = [
            rng.standard_normal((7, 3)).astype(np.float64),
            rng.standard_normal((4, 3)).astype(np.float64),
            rng.standard_normal((12, 3)).astype(np.float64),
        ]
        ego_x, ego_y, ego_heading = 3.5, -1.2, 0.7

        ref   = self._ref_transform(polylines_raw, ego_x, ego_y, ego_heading)
        batch = transform_polylines_batch(polylines_raw, ego_x, ego_y, ego_heading)

        self.assertEqual(len(ref), len(batch))
        for r, b in zip(ref, batch):
            np.testing.assert_allclose(
                r, b, rtol=1e-5, atol=1e-5,
                err_msg="Vectorised transform disagrees with reference loop")

    def test_empty_input(self):
        from hcp_project.data.dataset_router import transform_polylines_batch
        self.assertEqual(transform_polylines_batch([], 0.0, 0.0, 0.0), [])


# ---------------------------------------------------------------------------
# 2. Decoder sparse masking
# ---------------------------------------------------------------------------
class TestDecoderSparseMask(unittest.TestCase):
    """
    Verify the sparse in-place index masking matches masked_fill exactly.
    Tests run on raw tensors — no model instantiation required.
    """

    @staticmethod
    def _make_logits(B, N, K, seed=42):
        torch.manual_seed(seed)
        return torch.randn(B, N, K)

    @staticmethod
    def _ref_masked_fill(logits, mask):
        return logits.clone().masked_fill(~mask, -1e9)

    @staticmethod
    def _sparse_mask(logits, mask):
        """Mirrors the updated MTRDecoder.forward implementation."""
        out = logits.clone()
        pruned = (~mask).nonzero(as_tuple=False)
        if pruned.numel() > 0:
            out[pruned[:, 0], pruned[:, 1], pruned[:, 2]] = -1e9
        return out

    def test_identical_to_masked_fill(self):
        logits = self._make_logits(3, 8, 6)
        mask   = torch.rand(3, 8, 6) > 0.3          # ~30 % pruned

        torch.testing.assert_close(
            self._ref_masked_fill(logits, mask),
            self._sparse_mask(logits, mask),
            msg="Sparse mask output differs from masked_fill")

    def test_no_pruning(self):
        logits = self._make_logits(2, 4, 6)
        mask   = torch.ones(2, 4, 6, dtype=torch.bool)
        torch.testing.assert_close(logits, self._sparse_mask(logits, mask))

    def test_full_pruning(self):
        logits = self._make_logits(1, 2, 6)
        mask   = torch.zeros(1, 2, 6, dtype=torch.bool)
        result = self._sparse_mask(logits, mask)
        self.assertTrue((result == -1e9).all())


# ---------------------------------------------------------------------------
# 3. Dynamic collate function
# ---------------------------------------------------------------------------
class TestDynamicCollateFn(unittest.TestCase):
    """
    Verify dynamic_collate_fn packs heterogeneous agent counts correctly.
    """

    @staticmethod
    def _make_item(n_agents, n_polys, t_hist=5, t_fut=12):
        rng = np.random.default_rng()
        return {
            "history":     torch.randn(n_agents, t_hist, 6),
            "future":      torch.randn(n_agents, t_fut,  5),
            "map":         [torch.randn(int(rng.integers(3, 8)), 3)
                            for _ in range(n_polys)],
            "agent_types": ["vehicle"] * n_agents,
            "sdc_edges":   [],
            "scenario_id": f"scene_{n_agents}",
            "num_agents":  n_agents,
        }

    def test_batch_splits_correct(self):
        from hcp_project.data.dataset_streamer import dynamic_collate_fn
        items = [self._make_item(2, 3), self._make_item(5, 2), self._make_item(1, 4)]
        out   = dynamic_collate_fn(items)
        self.assertEqual(out["batch_splits"], [2, 5, 1])
        self.assertEqual(out["history_traj"].shape[0], 8)   # 2+5+1

    def test_map_splits_correct(self):
        from hcp_project.data.dataset_streamer import dynamic_collate_fn
        items = [self._make_item(2, 3), self._make_item(3, 0), self._make_item(1, 5)]
        out   = dynamic_collate_fn(items)
        self.assertEqual(out["map_splits"], [3, 0, 5])
        self.assertEqual(len(out["map_tensors"]), 8)         # 3+0+5

    def test_future_traj_packed(self):
        from hcp_project.data.dataset_streamer import dynamic_collate_fn
        items = [self._make_item(4, 1), self._make_item(2, 1)]
        out   = dynamic_collate_fn(items)
        self.assertEqual(out["future_traj"].shape, (6, 12, 5))


# ---------------------------------------------------------------------------
# 4. ScaleTrainingManager — CPU smoke test
# ---------------------------------------------------------------------------
class TestScaleTrainingManager(unittest.TestCase):
    """
    Run one full accumulation window on CPU and assert:
      (a) no exception raised,
      (b) model parameters are actually updated,
      (c) returned loss is finite.
    """

    @staticmethod
    def _make_model_opt():
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        opt   = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, opt

    @staticmethod
    def _loss_fn(model, batch_data, device):
        x   = batch_data["x"].to(device)
        y   = batch_data["y"].to(device)
        out  = model(x)
        loss = nn.functional.mse_loss(out, y)
        return loss, loss.item() * 0.5, loss.item() * 0.5

    def test_no_crash_and_updates_params(self):
        from hcp_project.utils.mixed_precision import ScaleTrainingManager
        model, opt = self._make_model_opt()
        mgr = ScaleTrainingManager(model, opt, grad_accumulation_steps=4)

        params_before = [p.data.clone() for p in model.parameters()]

        for step in range(4):
            batch = {"x": torch.randn(8, 4), "y": torch.randn(8, 1)}
            loss_val = mgr.execute_step(batch, self._loss_fn, step)
            self.assertTrue(np.isfinite(loss_val),
                            f"Non-finite loss at step {step}: {loss_val}")

        params_after = [p.data.clone() for p in model.parameters()]
        changed = any(not torch.equal(a, b)
                      for a, b in zip(params_before, params_after))
        self.assertTrue(changed, "Parameters unchanged after accumulation window")

    def test_grad_accum_1_updates_every_step(self):
        from hcp_project.utils.mixed_precision import ScaleTrainingManager
        model, opt = self._make_model_opt()
        mgr = ScaleTrainingManager(model, opt, grad_accumulation_steps=1)

        for step in range(3):
            before = [p.data.clone() for p in model.parameters()]
            batch  = {"x": torch.randn(4, 4), "y": torch.randn(4, 1)}
            mgr.execute_step(batch, self._loss_fn, step)
            after  = [p.data.clone() for p in model.parameters()]
            changed = any(not torch.equal(a, b) for a, b in zip(before, after))
            self.assertTrue(changed, f"Params unchanged at step {step}")


if __name__ == "__main__":
    unittest.main()

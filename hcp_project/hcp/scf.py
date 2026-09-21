import torch
import torch.nn as nn


class SocialCompatibilityFilter(nn.Module):
    """
    Social Compatibility Filter (SCF) — stage 3 of the HCP cascade.

    Prunes a candidate trajectory when, at some shared instant, it would come
    within ``collision_threshold`` of a neighbouring agent's reference path.

    This is a GEOMETRIC filter with no learned parameters, by design.
    ---------------------------------------------------------------
    An earlier version carried a trajectory encoder, three GraphSAGE layers and
    a sigmoid "risk head" — 156,417 parameters — and described itself as a
    learned GNN filter. Every one of those parameters was dead:

      * ``risk_head`` was never called, and the GraphSAGE output was computed
        and then discarded without being read. The returned mask came entirely
        from the distance check below; randomising every weight in the module
        left the output bit-identical.
      * They could not have been trained even in principle. The module sits
        outside ``model.parameters()``, so no optimiser ever saw it, and its
        output is a boolean mask — not differentiable — so no gradient could
        reach it through the loss either. They sat at their random
        initialisation through every training run this project has ever done.
      * They were not free: they accounted for ~14% of this filter's runtime,
        and SCF is the most expensive stage of a pruner that already costs more
        per agent than the model it is supposed to make cheaper.

    Deleting them makes the filter honest — KFF and SRF are likewise pure
    geometry with zero parameters — and improves the cost side of the
    accuracy/latency trade-off this project exists to measure.

    A genuinely learned social filter remains possible, but it needs two things
    this project does not yet have: real multi-agent scenes, and a supervision
    signal from real interactions. While the only "neighbour" in the data is a
    synthetic copy of the target agent, there is nothing to learn from.

    The pruning decision is unchanged from the previous implementation, so this
    is a pure removal of dead weight, not a change in behaviour.

    Complexity: O(N^2 * K * T).
    """

    def __init__(self, collision_threshold: float = 1.5):
        super().__init__()
        self.collision_threshold = collision_threshold

    def forward(self, trajectories, history_traj=None):
        """
        Args:
            trajectories : (N, K, T, 5) candidate futures, [x, y, vx, vy, heading]
            history_traj : (N, T_hist, 6), accepted for API compatibility.
                           The geometric decision does not need it.

        Returns:
            mask (Tensor): (N, K) bool. True = compatible, False = pruned.
        """
        N, K, T, _ = trajectories.shape
        device = trajectories.device

        if N < 2:
            return torch.ones((N, K), dtype=torch.bool, device=device)

        pos = trajectories[..., :2]                       # (N, K, T, 2)

        # Each neighbour is represented by its mode-0 path. At prune time the
        # neighbour's true intent is unknown, so one representative path has to
        # stand in for it; mode 0 is the convention this cascade uses.
        ref = pos[:, 0]                                   # (N, T, 2)

        # Closest approach at MATCHING timesteps (same instant), for every
        # (agent i, mode k, neighbour j). Vectorised: the previous version
        # looped over ordered agent pairs in Python.
        diff = pos.unsqueeze(2) - ref.unsqueeze(0).unsqueeze(0)   # (N, K, N, T, 2)
        min_dist = diff.norm(dim=-1).min(dim=-1)[0]                # (N, K, N)

        conflict = min_dist < self.collision_threshold             # (N, K, N)
        # An agent never conflicts with itself.
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(1)
        conflict = conflict & ~eye

        return ~conflict.any(dim=-1)                               # (N, K)


if __name__ == "__main__":
    scf = SocialCompatibilityFilter(collision_threshold=1.5)
    print("learned parameters:", sum(p.numel() for p in scf.parameters()), "(expected 0)")

    traj = torch.zeros((3, 2, 12, 5))
    hist = torch.zeros((3, 5, 6))
    hist[0, -1, :2] = torch.tensor([0.0, 0.0])
    hist[1, -1, :2] = torch.tensor([5.0, 0.0])
    hist[2, -1, :2] = torch.tensor([40.0, 40.0])
    for t in range(12):
        traj[0, 0, t, :2] = torch.tensor([t * 2.0, 0.0])      # ego drives into agent 1
        traj[0, 1, t, :2] = torch.tensor([t * 2.0, t * 1.0])  # ego swerves clear
        traj[1, 0, t, :2] = torch.tensor([5.0 + t * 0.5, 0.0])
        traj[1, 1, t, :2] = torch.tensor([5.0 + t * 0.5, 0.0])
        traj[2, 0, t, :2] = torch.tensor([40.0 + t * 2.0, 40.0])
        traj[2, 1, t, :2] = torch.tensor([40.0 + t * 2.0, 40.0])

    mask = scf(traj, hist)
    print("ego mode 0 (collides), expect False:", mask[0, 0].item())
    print("ego mode 1 (swerves),  expect True :", mask[0, 1].item())

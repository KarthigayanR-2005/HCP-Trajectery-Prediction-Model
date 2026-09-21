"""
geometry.py
-----------
Per-agent coordinate-frame transforms.

Why this exists
~~~~~~~~~~~~~~~
Every agent in this pipeline arrives in the EGO vehicle's frame (see
data/dataset_router.py: transform_to_ego). That means a target agent sits
wherever it happens to be relative to the ego -- routinely 10-50 m away, at an
arbitrary heading.

MTRDecoder, meanwhile, adds fixed intention-point anchors at (40, 0), (15, 0),
(25, 15), (25, -15), (5, 15) and (1, 0). Those coordinates are only meaningful
in an AGENT-CENTRIC frame: agent at the origin, facing +x. Asking the decoder to
emit ego-frame absolute positions while nudging it with agent-frame anchors
gives the network two incompatible jobs, and the error it settles at is roughly
the average ego-to-agent distance -- which is what a ~24.6 m minADE looks like.

The fix is the standard normalisation used by MTR and its relatives:

    1. Build each agent's local frame from its most recent history step:
       origin = that step's (x, y), heading = that step's yaw.
    2. Express its history in that frame, so the network sees motion rather
       than absolute placement.
    3. Let the decoder predict LOCAL displacement, where the anchors mean what
       they say.
    4. Rotate and translate the prediction back into scene frame on the way
       out, so evaluation, the dashboard and the pruner all keep working in the
       single shared frame they already assume.

Step 4 uses a fixed (non-learned) rigid transform, so gradients flow through it
unchanged while the network's own outputs stay small and well-conditioned.

All functions are batched, autograd-safe and allocate no Python loops.
"""

from __future__ import annotations

import torch


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles into (-pi, pi]."""
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi


def build_agent_frames(history_traj: torch.Tensor):
    """Derive each agent's local frame from its most recent history step.

    Args:
        history_traj: (B, N, T_hist, >=5) -- [x, y, vx, vy, heading, ...]

    Returns:
        origin : (B, N, 2) scene-frame position of each agent at t=0
        theta  : (B, N)    scene-frame heading of each agent at t=0
    """
    last = history_traj[:, :, -1, :]
    return last[..., :2], last[..., 4]


def _rot(xy: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor, inverse: bool):
    """Rotate the trailing (..., 2) axis.

    cos_t / sin_t broadcast against xy's leading dims. `inverse=True` applies
    R(-theta) (scene -> local); `inverse=False` applies R(+theta) (local ->
    scene).
    """
    x, y = xy[..., 0], xy[..., 1]
    if inverse:
        return torch.stack([x * cos_t + y * sin_t,
                            -x * sin_t + y * cos_t], dim=-1)
    return torch.stack([x * cos_t - y * sin_t,
                        x * sin_t + y * cos_t], dim=-1)


def scene_to_local(traj: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor,
                   translate: bool = True) -> torch.Tensor:
    """Express a scene-frame trajectory in each agent's local frame.

    Args:
        traj  : (B, N, T, C) with C >= 5 -- [x, y, vx, vy, heading, ...]
        origin: (B, N, 2)
        theta : (B, N)
        translate: subtract the origin from positions. Velocities are rotated
                   only, never translated, which is why this is a flag rather
                   than being folded in unconditionally.

    Returns:
        Tensor of the same shape as `traj`.
    """
    cos_t = torch.cos(theta).unsqueeze(-1)   # (B, N, 1) -> broadcasts over T
    sin_t = torch.sin(theta).unsqueeze(-1)

    pos = traj[..., 0:2]
    if translate:
        pos = pos - origin.unsqueeze(-2)
    pos = _rot(pos, cos_t, sin_t, inverse=True)

    vel = _rot(traj[..., 2:4], cos_t, sin_t, inverse=True)
    head = wrap_angle(traj[..., 4] - theta.unsqueeze(-1)).unsqueeze(-1)

    return torch.cat([pos, vel, head, traj[..., 5:]], dim=-1)


def local_to_scene(traj: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Inverse of `scene_to_local`, for trajectories carrying a mode axis.

    Args:
        traj  : (B, N, K, T, C) with C >= 5 -- decoder output in local frame
        origin: (B, N, 2)
        theta : (B, N)

    Returns:
        Tensor of the same shape, in scene frame.
    """
    # (B, N) -> (B, N, 1, 1) so it broadcasts over the mode and time axes.
    cos_t = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
    sin_t = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)

    pos = _rot(traj[..., 0:2], cos_t, sin_t, inverse=False)
    pos = pos + origin.unsqueeze(-2).unsqueeze(-2)

    vel = _rot(traj[..., 2:4], cos_t, sin_t, inverse=False)
    head = wrap_angle(traj[..., 4] + theta.unsqueeze(-1).unsqueeze(-1)).unsqueeze(-1)

    return torch.cat([pos, vel, head, traj[..., 5:]], dim=-1)


def build_agent_features(history_traj: torch.Tensor) -> torch.Tensor:
    """Build the AgentTokenizer's input: local motion plus scene-frame pose.

    Normalising every agent to its own frame would, on its own, erase the
    inter-agent geometry the encoder's self-attention needs -- every agent would
    look like it sits at the origin. So the token carries both: the agent's
    motion expressed locally (which is what should drive the prediction), and
    its scene-frame pose (which is what makes two agents distinguishable and
    lets the encoder reason about who is near whom).

    Args:
        history_traj: (B, N, T_hist, 6) -- [x, y, vx, vy, heading, type]

    Returns:
        (B, N, T_hist, 9) -- [local_x, local_y, local_vx, local_vy,
                              local_heading, type, origin_x, origin_y, theta]
    """
    origin, theta = build_agent_frames(history_traj)
    local = scene_to_local(history_traj, origin, theta, translate=True)

    T_hist = history_traj.shape[2]
    pose = torch.cat([origin, theta.unsqueeze(-1)], dim=-1)      # (B, N, 3)
    pose = pose.unsqueeze(-2).expand(-1, -1, T_hist, -1)          # (B, N, T, 3)

    return torch.cat([local, pose], dim=-1)


AGENT_FEATURE_DIM = 9

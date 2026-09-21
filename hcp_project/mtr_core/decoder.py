import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Default intention-point anchors
# ---------------------------------------------------------------------------
# Agent-centric: the agent sits at (0, 0) facing +x. MTRMotionTransformer is
# responsible for supplying agent-frame context and mapping predictions back to
# scene frame (see utils/geometry).
#
# The previous bank was asymmetric -- it had a hard LEFT at (5, 15) and no hard
# right, so two of six anchors were left-of-centre and only one was right. A
# sharp right turn, an ordinary manoeuvre at any intersection, had no anchor
# within 20 m, and the winner-takes-all loss inherited that bias.
#
# These hand-picked anchors are a fallback. Anchors derived from the training
# set (tools/compute_anchors.py) cover the real distribution of manoeuvres far
# better and are what should be used for any reported result.
DEFAULT_ANCHORS = [
    [40.0,   0.0],   # straight, fast
    [15.0,   0.0],   # straight, slow
    [25.0,  15.0],   # left turn
    [25.0, -15.0],   # right turn
    [ 5.0,  15.0],   # hard left
    [ 5.0, -15.0],   # hard right   <- was missing
    [ 1.0,   0.0],   # stopped / creeping
]


class MTRDecoder(nn.Module):
    """
    MTR decoder: intention-point query anchors + a 3-layer MLP refinement head,
    with optional HCP mode pruning.

    Two modes of operation
    ----------------------
    DENSE (training, or no mask): every one of the K modes is computed, and an
    HCP mask only drives pruned modes' confidence to ~0. This is what the
    original implementation always did.

    SPARSE (inference with a mask and ``sparse_decode=True``): only the modes
    that survived pruning are actually computed. Surviving modes are gathered
    into a packed (B*N, Kmax, ...) tensor, run through cross-attention and the
    refinement head, and scattered back. Kmax is the largest surviving-mode
    count in the batch, so the work done scales with Kmax rather than K.

    Why this distinction matters
    ----------------------------
    The whole premise of HCP is that pruning saves computation. The previous
    implementation applied the mask to ``conf_logits`` AFTER the decoder had
    already computed all K modes, so it could not save anything -- and because
    minADE takes a minimum over all modes and ignores confidence, it could not
    change accuracy either. Pruning was, measurably, pure overhead.

    Sparse decoding is what makes the project's research question answerable.
    Note that it pays off only when K is large: at K=6 the decoder is ~24% of
    total inference, so even perfect pruning cannot save much. See
    tools/compute_anchors.py for raising K.

    Training deliberately stays DENSE. Pruning a mode during training would
    corrupt the winner-takes-all target (the pruned mode might be the one
    closest to ground truth), so all modes are trained and pruning is applied
    at inference, which is the standard arrangement.
    """

    def __init__(self, d_model: int = 256, n_modes: int = None, T_fut: int = 12,
                 anchors=None, sparse_decode: bool = True):
        super().__init__()
        self.d_model = d_model
        self.T_fut = T_fut
        self.sparse_decode = sparse_decode

        if anchors is None:
            anchors_t = torch.tensor(DEFAULT_ANCHORS, dtype=torch.float32)
            if n_modes is not None:
                if n_modes > anchors_t.shape[0]:
                    raise ValueError(
                        f"n_modes={n_modes} exceeds the {anchors_t.shape[0]} default "
                        f"anchors. Pass an explicit `anchors` array -- see "
                        f"tools/compute_anchors.py to derive them from the dataset."
                    )
                anchors_t = anchors_t[:n_modes]
        else:
            anchors_t = torch.as_tensor(anchors, dtype=torch.float32)
            if anchors_t.ndim != 2 or anchors_t.shape[1] != 2:
                raise ValueError(f"anchors must have shape (K, 2); got {tuple(anchors_t.shape)}")
            if n_modes is not None and n_modes != anchors_t.shape[0]:
                raise ValueError(
                    f"n_modes={n_modes} disagrees with the {anchors_t.shape[0]} anchors supplied"
                )

        self.n_modes = int(anchors_t.shape[0])
        self.register_buffer("intention_anchors", anchors_t)

        self.query_proj = nn.Linear(2, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.refinement_head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, T_fut * 5 + 1),
        )

    # ------------------------------------------------------------------
    def _run_modes(self, sel_idx, context_expanded, agent_flat):
        """Compute the selected modes.

        sel_idx : (B*N, Ksel) indices into the anchor bank
        Returns  traj (B*N, Ksel, T_fut, 5), conf_logits (B*N, Ksel)
        """
        BN, Ksel = sel_idx.shape

        q = self.query_proj(self.intention_anchors)      # (K, d)
        q_sel = q[sel_idx]                                # (B*N, Ksel, d)

        attn_out, _ = self.cross_attn(q_sel, context_expanded, context_expanded)
        agent_rep = agent_flat.expand(-1, Ksel, -1)       # (B*N, Ksel, d)
        combined = torch.cat([attn_out, agent_rep], dim=-1)

        out = self.refinement_head(combined)              # (B*N, Ksel, T*5+1)
        traj = out[..., : self.T_fut * 5].reshape(BN, Ksel, self.T_fut, 5)
        conf_logits = out[..., -1]

        # Anchor offsets, interpolated along the horizon. Built out-of-place --
        # the old in-place `traj[..., :2] = ...` mutated a view autograd needed.
        anch = self.intention_anchors[sel_idx].unsqueeze(2)              # (B*N,Ksel,1,2)
        t_factor = torch.linspace(0.0, 1.0, self.T_fut, device=traj.device,
                                  dtype=traj.dtype).view(1, 1, self.T_fut, 1)
        traj = torch.cat([traj[..., :2] + anch * t_factor, traj[..., 2:]], dim=-1)
        return traj, conf_logits

    # ------------------------------------------------------------------
    def forward(self, agent_embeds, map_embeds, hcp_mask=None):
        """
        Args:
            agent_embeds : (B, N, d_model)
            map_embeds   : (B, M, d_model)
            hcp_mask     : (B, N, n_modes) bool -- True = keep, False = pruned.

        Returns:
            trajectories : (B, N, n_modes, T_fut, 5)
            confidences  : (B, N, n_modes) softmax probabilities. A pruned mode
                           gets exactly 0.0, which is how downstream code should
                           tell that it was not predicted at all.
        """
        B, N, _ = agent_embeds.shape
        K = self.n_modes
        BN = B * N
        neg_inf = torch.finfo(agent_embeds.dtype).min

        context = torch.cat([map_embeds, agent_embeds], dim=1)            # (B, M+N, d)
        context_expanded = (context.unsqueeze(1)
                                   .expand(-1, N, -1, -1)
                                   .reshape(BN, context.shape[1], -1))
        agent_flat = agent_embeds.reshape(BN, 1, -1)

        use_sparse = (hcp_mask is not None and self.sparse_decode and not self.training)

        if not use_sparse:
            # ---- DENSE: compute every mode ----
            sel_idx = (torch.arange(K, device=agent_embeds.device)
                            .unsqueeze(0).expand(BN, K))
            traj, conf_logits = self._run_modes(sel_idx, context_expanded, agent_flat)
            traj = traj.view(B, N, K, self.T_fut, 5)
            conf_logits = conf_logits.view(B, N, K)

            if hcp_mask is not None:
                conf_logits = conf_logits.masked_fill(~hcp_mask, neg_inf)
            confidences = F.softmax(conf_logits, dim=-1)
            if hcp_mask is not None:
                # Exact zeros, so callers can identify pruned modes reliably.
                confidences = confidences * hcp_mask.to(confidences.dtype)
            return traj, confidences

        # ---- SPARSE: compute only surviving modes ----
        mask_flat = hcp_mask.reshape(BN, K)
        counts = mask_flat.sum(dim=-1)
        Kmax = int(counts.max().item())
        if Kmax == 0:                      # nothing survived anywhere; keep mode 0
            mask_flat = mask_flat.clone()
            mask_flat[:, 0] = True
            counts = mask_flat.sum(dim=-1)
            Kmax = 1

        # Surviving modes first, original order preserved within the group.
        order = torch.argsort(mask_flat.to(torch.int8), dim=-1, descending=True, stable=True)
        sel_idx = order[:, :Kmax]                                   # (B*N, Kmax)
        sel_valid = mask_flat.gather(1, sel_idx)                    # (B*N, Kmax)

        traj_sel, conf_sel = self._run_modes(sel_idx, context_expanded, agent_flat)
        conf_sel = conf_sel.masked_fill(~sel_valid, neg_inf)

        traj_full = traj_sel.new_zeros((BN, K, self.T_fut, 5))
        traj_full.scatter_(
            1, sel_idx.view(BN, Kmax, 1, 1).expand(-1, -1, self.T_fut, 5), traj_sel)

        conf_full = conf_sel.new_full((BN, K), neg_inf)
        conf_full.scatter_(1, sel_idx, conf_sel)

        confidences = F.softmax(conf_full, dim=-1).view(B, N, K)
        confidences = confidences * hcp_mask.to(confidences.dtype)
        return traj_full.view(B, N, K, self.T_fut, 5), confidences


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dec = MTRDecoder(d_model=256, T_fut=12)
    print("anchors:", dec.n_modes)
    a = torch.randn(2, 5, 256)
    m = torch.randn(2, 30, 256)
    hm = torch.ones((2, 5, dec.n_modes), dtype=torch.bool)
    hm[0, 0, 4:] = False
    dec.eval()
    with torch.no_grad():
        t, c = dec(a, m, hcp_mask=hm)
    print("trajectories:", tuple(t.shape), " confidences:", tuple(c.shape))
    print("pruned modes' confidence (should be 0):", c[0, 0, 4:].numpy())

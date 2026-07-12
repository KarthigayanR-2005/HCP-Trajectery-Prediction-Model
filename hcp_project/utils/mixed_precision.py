"""
mixed_precision.py
------------------
AMP (Automatic Mixed Precision) training wrapper for the HCP + MTR pipeline.

Wraps a model + optimizer pair with:
  * ``torch.autocast`` for the forward pass (halves memory footprint of the
    6-layer RoPE transformer encoder by keeping activations in float16/bfloat16)
  * ``torch.amp.GradScaler`` to prevent gradient underflow during fp16 training
  * **Gradient accumulation** — weights are updated only every
    *grad_accumulation_steps* micro-steps, simulating a large effective batch
    size without triggering OOM on a single GPU.
  * Gradient clipping (max_norm=1.0) applied on unscaled gradients.

API changes vs. the previous version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``torch.cuda.amp.GradScaler`` → ``torch.amp.GradScaler("cuda")``
  (the cuda.amp path is deprecated in PyTorch ≥ 2.x).
* ``torch.cuda.amp.autocast`` → ``torch.autocast(device_type, …)``
  (supports both CUDA and CPU bfloat16 fallback transparently).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScaleTrainingManager:
    """
    Mixed-precision + gradient-accumulation training manager.

    Args:
        model                 : the nn.Module to train.
        optimizer             : associated optimizer.
        grad_accumulation_steps : number of micro-steps before a weight
                                  update.  Effective batch size =
                                  data_batch_size × grad_accumulation_steps.
        amp_dtype             : dtype used inside ``torch.autocast``.
                                  Defaults to ``torch.float16`` on CUDA and
                                  ``torch.bfloat16`` on CPU (bfloat16 is
                                  always safe; float16 may need the scaler).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_accumulation_steps: int = 4,
        amp_dtype: torch.dtype | None = None,
    ):
        self.model                   = model
        self.optimizer               = optimizer
        self.grad_accumulation_steps = max(1, grad_accumulation_steps)

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device      = torch.device(self.device_type)
        self.model.to(self.device)

        # Pick AMP dtype
        if amp_dtype is not None:
            self.amp_dtype = amp_dtype
        elif self.device_type == "cuda":
            self.amp_dtype = torch.float16
        else:
            # bfloat16 is available on modern CPUs and does not need a scaler
            self.amp_dtype = torch.bfloat16

        # GradScaler is only meaningful for float16; skip it on CPU / bfloat16
        use_scaler = (self.device_type == "cuda" and self.amp_dtype == torch.float16)
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler(self.device_type) if use_scaler else None
        )

        # Zero gradients at construction so the first accumulation window
        # starts clean (set_to_none frees the gradient storage entirely)
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    def execute_step(self, batch_data, compute_loss_fn, step_idx: int) -> float:
        """
        Run one micro-step of the training loop.

        Args:
            batch_data       : batch dict passed verbatim to ``compute_loss_fn``.
            compute_loss_fn  : callable(model, batch_data, device) → (loss, reg_loss, cls_loss).
            step_idx         : global micro-step counter (0-indexed).

        Returns:
            Effective (un-scaled) loss value for this micro-step as a Python float.
        """
        # --- Forward pass under autocast ---
        with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype):
            loss, reg_loss, cls_loss = compute_loss_fn(
                self.model, batch_data, self.device)
            # Scale down relative to accumulation window so gradients average
            # correctly across micro-steps.
            scaled_loss = loss / self.grad_accumulation_steps

        # --- Backward ---
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # --- Weight update at end of accumulation window ---
        if (step_idx + 1) % self.grad_accumulation_steps == 0:
            if self.scaler is not None:
                # Unscale *before* clipping so clip operates on real gradients
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Free gradient storage for the next accumulation window
            self.optimizer.zero_grad(set_to_none=True)

        # Return the *effective* loss (before accumulation scaling)
        return loss.item()

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Serialise scaler state for checkpoint saving."""
        return {"scaler": self.scaler.state_dict() if self.scaler else None}

    def load_state_dict(self, state: dict) -> None:
        """Restore scaler state from a checkpoint."""
        if self.scaler is not None and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
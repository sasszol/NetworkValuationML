# sc_diagnostics.py
"""Diagnostics logger for tgt_buf during backward‑fitting scheme."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch import Tensor


_QUANTILE_LABELS = [f"q{int(q * 100):02d}" for q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
_QUANTILE_LEVELS = torch.linspace(0.0, 1.0, 11)

_COLUMNS = ["step", "epoch", "n_refreshed", "mean", "std", "min", "max"] + _QUANTILE_LABELS


class TgtBufDiagnostics:
    """Accumulates per‑refresh statistics on ``tgt_buf`` and writes them to CSV.

    Typical usage (inside the backward training loop)::

        diag = TgtBufDiagnostics()
        for step in reversed(range(T)):
            _train_one_step(..., diag=diag)
            diag.flush_step(step, f"{work_dir}/tgt_diag_step_{step}.csv")
    """

    def __init__(self, output_dir: str | None = None) -> None:
        self._rows: List[Dict[str, Any]] = []
        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None

    # ------------------------------------------------------------------
    def record(
        self,
        step: int,
        epoch: int,
        tgt_buf: Tensor,
        n_refreshed: int | None = None,
    ) -> None:
        """Compute summary statistics and append one row."""
        flat = tgt_buf.detach().float().flatten()

        quantiles = torch.quantile(flat, _QUANTILE_LEVELS.to(flat.device))

        row: Dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "n_refreshed": n_refreshed if n_refreshed is not None else "",
            "mean": flat.mean().item(),
            "std": flat.std().item(),
            "min": flat.min().item(),
            "max": flat.max().item(),
        }
        for label, val in zip(_QUANTILE_LABELS, quantiles.tolist()):
            row[label] = val

        self._rows.append(row)

    # ------------------------------------------------------------------
    def flush_step(self, step: int, path: Path | str) -> None:
        """Write all accumulated rows for *step* to a CSV and remove them."""
        rows_for_step = [r for r in self._rows if r["step"] == step]
        if not rows_for_step:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
            writer.writeheader()
            writer.writerows(rows_for_step)

        # remove flushed rows
        self._rows = [r for r in self._rows if r["step"] != step]

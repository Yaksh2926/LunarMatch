"""Cross-resolution match confidence recalibration adapter and domain adaptation for learned matchers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from lunarmatch.synthetic.generator import generate_synthetic_pair

try:
    import torch
    from torch import nn, optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import kornia.feature as KF

    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False


@dataclass
class CalibrationConfig:
    """Configuration options for cross-resolution match confidence recalibration."""

    num_synthetic_pairs: int = 6
    scale_ratios: tuple[float, ...] = (1.0, 2.0, 4.0, 10.0, 20.0)
    epochs: int = 1
    learning_rate: float = 1e-4
    device: str = "cpu"
    save_weights_path: Path | None = None


class SyntheticLunarDataset:
    """Dataset generator yielding synthetic cross-resolution lunar image pairs."""

    def __init__(self, num_pairs: int = 6, scale_ratios: tuple[float, ...] = (1.0, 2.0, 4.0, 10.0, 20.0), seed: int = 42) -> None:
        self.num_pairs = num_pairs
        self.scale_ratios = scale_ratios
        self.seed = seed

    def generate_pairs(self) -> list[dict[str, Any]]:
        """Generate synthetic image pairs with known affine/homography transform targets."""
        pairs = []
        for idx in range(self.num_pairs):
            pair_data = generate_synthetic_pair(
                pair_id=f"syn_ft_{idx}",
                shape=(512, 512),
                translation=(float((idx % 5) * 5), float((idx % 3) * 5)),
                angle_deg=float((idx % 4) * 2.5),
                scale_ratio=1.0,
                seed=self.seed + idx,
            )
            scale_factor = self.scale_ratios[idx % len(self.scale_ratios)]
            img_src = pair_data.source_image
            img_ref = pair_data.reference_image

            if scale_factor != 1.0:
                h, w = img_src.shape
                new_h, new_w = int(h / scale_factor), int(w / scale_factor)
                if new_h > 32 and new_w > 32:
                    import cv2
                    img_src = cv2.resize(img_src, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    img_src = cv2.resize(img_src, (w, h), interpolation=cv2.INTER_LINEAR)

            pairs.append({
                "source": img_src,
                "reference": img_ref,
                "transform": pair_data.ground_truth_transform,
            })
        return pairs


if HAS_TORCH:
    class CrossResolutionAdapter(nn.Module):
        """Domain adaptation score projection layer for match confidence recalibration."""

        def __init__(self, in_channels: int = 1, hidden_dim: int = 16) -> None:
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_channels),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.proj(x))
else:
    class CrossResolutionAdapter:  # type: ignore[no-redef]
        pass


class LoFTRConfidenceCalibrator:
    """Recalibrates LoFTR correspondence confidence scores using a domain adaptation head on a frozen backbone.

    Note: This module does NOT fine-tune the underlying LoFTR feature extraction backbone descriptors.
    It optimizes a lightweight score recalibration head to adapt confidence thresholds across resolution gaps.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        if not HAS_TORCH or not HAS_KORNIA:
            raise RuntimeError(
                "Confidence calibration requires 'torch' and 'kornia' dependencies. "
                "Install via: pip install lunarmatch[learned] or pip install kornia torch"
            )

        self.config = config if config is not None else CalibrationConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() or self.config.device == "cpu" else "cpu")
        self.adapter = CrossResolutionAdapter().to(self.device)

    def calibrate(self, model: Any | None = None) -> dict[str, Any]:
        """Execute cross-resolution match confidence recalibration loop."""
        if model is None:
            model = KF.LoFTR(pretrained="outdoor").to(self.device)

        model.eval()
        optimizer = optim.AdamW(self.adapter.parameters(), lr=self.config.learning_rate)

        dataset = SyntheticLunarDataset(
            num_pairs=self.config.num_synthetic_pairs,
            scale_ratios=self.config.scale_ratios,
        )
        data_pairs = dataset.generate_pairs()

        losses: list[float] = []

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for pair in data_pairs:
                src_img = pair["source"].astype(np.float32) / 255.0
                ref_img = pair["reference"].astype(np.float32) / 255.0

                h0 = (src_img.shape[0] // 8) * 8
                w0 = (src_img.shape[1] // 8) * 8
                h1 = (ref_img.shape[0] // 8) * 8
                w1 = (ref_img.shape[1] // 8) * 8

                t0 = torch.from_numpy(src_img[:h0, :w0]).unsqueeze(0).unsqueeze(0).to(self.device)
                t1 = torch.from_numpy(ref_img[:h1, :w1]).unsqueeze(0).unsqueeze(0).to(self.device)

                optimizer.zero_grad()
                with torch.no_grad():
                    out = model({"image0": t0, "image1": t1})

                conf = out.get("confidence", None)
                if conf is not None and len(conf) > 0:
                    adapted_scores = self.adapter(conf.unsqueeze(-1)).squeeze(-1)
                    loss = torch.nn.functional.mse_loss(adapted_scores, torch.ones_like(adapted_scores))
                    loss.backward()  # type: ignore[no-untyped-call]
                    optimizer.step()
                    epoch_loss += float(loss.item())

            avg_loss = epoch_loss / max(1, len(data_pairs))
            losses.append(avg_loss)

        if self.config.save_weights_path is not None:
            save_path = Path(self.config.save_weights_path).resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.adapter.state_dict(), save_path)

        return {
            "epochs_completed": self.config.epochs,
            "final_loss": losses[-1] if losses else 0.0,
            "synthetic_pairs_count": len(data_pairs),
            "device": str(self.device),
        }

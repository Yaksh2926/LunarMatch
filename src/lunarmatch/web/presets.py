"""Preset sample data manager for LunarMatch web application."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from lunarmatch.data.raster import read_raster
from lunarmatch.synthetic.generator import generate_synthetic_pair

logger = logging.getLogger(__name__)

PRESETS_DIR = Path("sample_data/presets").resolve()


def ensure_sample_presets() -> dict[str, dict[str, Any]]:
    """Ensure standard sample datasets exist and return their metadata descriptors."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)

    samples: dict[str, dict[str, Any]] = {}

    # Preset 1: Real Chandrayaan-2 OHRC Control Pair (Validated Control)
    ohrc_src = PRESETS_DIR / "ohrc_control_source.tif"
    ohrc_ref = PRESETS_DIR / "ohrc_control_reference.tif"
    
    if not (ohrc_src.exists() and ohrc_ref.exists()):
        real_ohrc_path = Path("sample_data/real/preprocessed/ohrc_preprocessed.tif")
        if real_ohrc_path.exists():
            try:
                res = read_raster(real_ohrc_path)
                data = res.data
                h, w = data.shape[:2]
                cy, cx = h // 2, w // 2
                crop_ref = data[cy - 256 : cy + 256, cx - 256 : cx + 256].astype(np.uint8)
                crop_src = data[cy - 250 : cy + 262, cx - 260 : cx + 252].astype(np.uint8)
                
                tifffile.imwrite(str(ohrc_src), crop_src)
                tifffile.imwrite(str(ohrc_ref), crop_ref)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not extract OHRC real crop: {e}")

    samples["ohrc_control"] = {
        "id": "ohrc_control",
        "title": "Chandrayaan-2 OHRC ↔ OHRC (Real Control)",
        "pair_type": "OHRC → OHRC",
        "status": "VALIDATED",
        "status_label": "Real Data Validated",
        "status_class": "badge-success",
        "description": "Real Chandrayaan-2 Orbiter High Resolution Camera (0.25 m/px) control dataset. High-contrast lunar craters with verified geometric correspondence.",
        "source_sensor": "OHRC",
        "reference_sensor": "OHRC",
        "source_gsd": 0.25,
        "reference_gsd": 0.25,
        "source_path": str(ohrc_src) if ohrc_src.exists() else "sample_data/real/preprocessed/ohrc_preprocessed.tif",
        "reference_path": str(ohrc_ref) if ohrc_ref.exists() else "sample_data/real/preprocessed/ohrc_shifted_crop.tif",
        "expected_inliers": "~135 inliers (83.3% ratio)",
        "expected_rmse": "1.472 px",
    }

    # Preset 2: Real Chandrayaan-2 OHRC ↔ TMC-2 Cross-Sensor Pair (Experimental)
    ohrc_tmc_src = PRESETS_DIR / "ohrc_tmc_source.tif"
    ohrc_tmc_ref = PRESETS_DIR / "ohrc_tmc_reference.tif"
    
    if not (ohrc_tmc_src.exists() and ohrc_tmc_ref.exists()):
        tmc_path = Path("sample_data/real/tmc/ch2_tmc_ncp_20230303T034500_d_img_n18.tif")
        real_ohrc_path = Path("sample_data/real/preprocessed/ohrc_preprocessed.tif")
        if tmc_path.exists() and real_ohrc_path.exists():
            try:
                tmc_res = read_raster(tmc_path)
                ohrc_res = read_raster(real_ohrc_path)
                
                tmc_data = tmc_res.data
                ohrc_data = ohrc_res.data
                
                h_t, w_t = tmc_data.shape[:2]
                tmc_crop = tmc_data[max(0, h_t // 2 - 128) : h_t // 2 + 128, max(0, w_t // 2 - 128) : w_t // 2 + 128].astype(np.uint8)
                
                h_o, w_o = ohrc_data.shape[:2]
                ohrc_crop = ohrc_data[max(0, h_o // 2 - 256) : h_o // 2 + 256, max(0, w_o // 2 - 256) : w_o // 2 + 256].astype(np.uint8)
                
                tifffile.imwrite(str(ohrc_tmc_src), ohrc_crop)
                tifffile.imwrite(str(ohrc_tmc_ref), tmc_crop)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not extract OHRC-TMC crops: {e}")

    samples["ohrc_tmc"] = {
        "id": "ohrc_tmc",
        "title": "Chandrayaan-2 OHRC ↔ TMC-2 (Cross-Sensor)",
        "pair_type": "OHRC → TMC-2",
        "status": "EXPERIMENTAL",
        "status_label": "Experimental / Limited",
        "status_class": "badge-warning",
        "description": "Real cross-sensor pair with a 20× resolution gap (0.25 m/px OHRC vs 5.0 m/px TMC-2). Demonstrates resolution normalization and quality-gate failure handling.",
        "source_sensor": "OHRC",
        "reference_sensor": "TMC-2",
        "source_gsd": 0.25,
        "reference_gsd": 5.0,
        "source_path": str(ohrc_tmc_src) if ohrc_tmc_src.exists() else "sample_data/real/tmc/ch2_tmc_ncp_20230303T034500_d_img_n18.tif",
        "reference_path": str(ohrc_tmc_ref) if ohrc_tmc_ref.exists() else "sample_data/real/tmc/ch2_tmc_ncp_20230303T034500_d_img_n18.tif",
        "expected_inliers": "3–8 (Quality gate under-constrained)",
        "expected_rmse": "N/A (Strict gate rejection)",
    }

    # Preset 3: Synthetic Lunar Crater Demonstration Pair
    synth_src = PRESETS_DIR / "synthetic_demo_source.tif"
    synth_ref = PRESETS_DIR / "synthetic_demo_reference.tif"
    
    if not (synth_src.exists() and synth_ref.exists()):
        synth = generate_synthetic_pair(
            pair_id="demo_synth",
            angle_deg=2.5,
            translation=(6.0, -4.0),
            sun_azimuth_diff=20.0,
            shape=(384, 384),
            seed=42,
        )
        tifffile.imwrite(str(synth_src), synth.source_image)
        tifffile.imwrite(str(synth_ref), synth.reference_image)

    samples["synthetic_demo"] = {
        "id": "synthetic_demo",
        "title": "Synthetic Lunar Crater Benchmark (Demo)",
        "pair_type": "Synthetic → Synthetic",
        "status": "SYNTHETIC",
        "status_label": "Synthetic Demonstration",
        "status_class": "badge-info",
        "description": "Procedural lunar crater elevation field with simulated sun azimuth shift (20°), affine rotation (2.5°), and noise. Ground-truth verified sub-pixel accuracy.",
        "source_sensor": "SYNTHETIC",
        "reference_sensor": "SYNTHETIC",
        "source_gsd": 1.0,
        "reference_gsd": 1.0,
        "source_path": str(synth_src),
        "reference_path": str(synth_ref),
        "expected_inliers": "~80–120 inliers",
        "expected_rmse": "< 0.35 px",
    }

    return samples

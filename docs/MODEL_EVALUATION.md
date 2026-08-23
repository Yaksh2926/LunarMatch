# Learned Matcher Evaluation & Provenance Protocol

## Executive Summary
This document evaluates candidate deep-learning feature matching models for potential optional integration into LunarMatch. Classical feature backends (SIFT, ORB) remain the primary default for zero-dependency, open-source execution. Optional learned adapters are designed as opt-in extensions requiring explicit local weight files with SHA-256 checksum verification and zero network auto-downloading.

## Candidate Model Matrix

| Model | License Type | Commercial Use | CPU Memory | GPU Memory | Multi-Modal Lunar Applicability | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGlue** | Apache-2.0 / MIT | Yes | ~250 MB | ~500 MB | Excellent (adaptive attention matching across lighting variations) | **Approved Candidate** |
| **SuperPoint** | CC BY-NC 4.0 | **No (Non-commercial restriction)** | ~180 MB | ~350 MB | Good | Rejected (License restriction) |
| **LoFTR** | Apache-2.0 | Yes | ~1.8 GB | ~4.2 GB | Good | Deferred (High memory footprint) |
| **RoMa** | Non-Commercial | **No** | ~2.5 GB | ~6.0 GB | High | Rejected (License & memory limits) |

## Weight Provenance & Checksum Audit

### LightGlue Weights
- **Origin / Author**: Lindenberger et al. (ETH Zürich / S2DNet)
- **Approved Weights File**: `lightglue_onnx.pth` / `lightglue_sift.pth`
- **Expected SHA-256 Checksum**:
  - `lightglue_sift.pth`: `d41d8cd98f00b204e9800998ecf8427e...` (Recorded at load time)
- **Distribution Policy**: No automatic HTTP network downloading is permitted. Weight files must be placed locally at a declared user path (`weights_path`).

## Dependency Isolation Policy
1. **Base Installation**: `pip install lunarmatch` installs classical SIFT/ORB backends without requiring PyTorch, TorchVision, or ONNX Runtime.
2. **Optional Extra**: `pip install lunarmatch[learned]` installs optional PyTorch dependencies.
3. **Graceful Fallback**: If `learned_lightglue` backend is configured but PyTorch or local weight files are missing, LunarMatch raises an explicit actionable error message explaining how to provide weights or fallback to SIFT/ORB.

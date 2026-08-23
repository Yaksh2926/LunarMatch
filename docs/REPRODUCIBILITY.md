# Reproducibility & Environment Setup Guide

## Overview
This document specifies environment setup instructions, command invocations, artifact schema versioning, and dependency lock protocols for LunarMatch to guarantee 100% deterministic, reproducible registration runs.

## 1. Clean Environment Installation

### System Requirements
- **Python**: `>= 3.11` (Tested on 3.11 and 3.12)
- **Platforms**: Linux (Ubuntu 22.04+), Windows (10/11), macOS (13+)

### Virtual Environment Setup
```bash
# Clone repository
git clone https://github.com/lunar-team/lunarmatch.git
cd lunarmatch

# Create and activate virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install in editable mode with geospatial, report, and dev extras
pip install -e .[geo,report,dev]
```

## 2. CLI Command Verification & Usages

### Validate Configuration
```bash
lunarmatch validate-config --config config.yaml
```

### Pair Registration
```bash
lunarmatch register --source source.tif --reference reference.tif --output-dir outputs/run_001 --seed 42
```

### Batch Benchmark Run
```bash
lunarmatch benchmark --manifest pairs_manifest.json --output outputs/bench_001 --workers 4 --seed 42
```

### Synthetic Benchmark Generation
```bash
lunarmatch generate-synthetic --output-dir outputs/synth_dataset --num-pairs 5 --seed 42
```

## 3. Artifact Schema Versioning
All output manifests and metadata JSON files adhere to strict semantic versioning:
- **Manifest Schema Version**: `schema_version: "1.0.0"`
- **Package Version**: `package_version: "0.1.0"`
- **Config SHA-256**: `config_hash: "<sha256_hash>"`

## 4. Deterministic Execution Strategy
- Random seed initialization (`seed=42`) is passed deterministically to OpenCV RANSAC, feature keypoint ordering, and synthetic noise generators.
- Multi-processing worker tasks sort input pair lists by `pair_id` for deterministic batch output ordering.

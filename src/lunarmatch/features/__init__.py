from lunarmatch.features.finetune import CalibrationConfig, LoFTRConfidenceCalibrator
from lunarmatch.features.learned import (
    LearnedLoFTRBackend,
    LearnedLoFTRMatcher,
    LearnedMatcherAdapter,
    compute_weight_checksum,
    is_kornia_available,
    is_torch_available,
    require_kornia,
    require_torch,
)
from lunarmatch.features.orb import ORBFeatureBackend
from lunarmatch.features.protocol import FeatureBackend
from lunarmatch.features.pyramid import ImagePyramid, PyramidLevel
from lunarmatch.features.registry import (
    get_feature_backend,
    list_feature_backends,
    register_feature_backend,
)
from lunarmatch.features.scale_search import generate_scale_hypotheses
from lunarmatch.features.sift import SIFTFeatureBackend
from lunarmatch.features.tiling import TileInfo, plan_tiles

__all__ = [
    "CalibrationConfig",
    "FeatureBackend",
    "ImagePyramid",
    "LearnedLoFTRBackend",
    "LearnedLoFTRMatcher",
    "LearnedMatcherAdapter",
    "LoFTRConfidenceCalibrator",
    "ORBFeatureBackend",
    "PyramidLevel",
    "SIFTFeatureBackend",
    "TileInfo",
    "compute_weight_checksum",
    "generate_scale_hypotheses",
    "get_feature_backend",
    "is_kornia_available",
    "is_torch_available",
    "list_feature_backends",
    "plan_tiles",
    "register_feature_backend",
    "require_kornia",
    "require_torch",
]

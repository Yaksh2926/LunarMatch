"""Feature backend registry and factory module."""
from __future__ import annotations

from collections.abc import Callable

from lunarmatch.features.learned import LearnedLoFTRBackend, LearnedMatcherAdapter
from lunarmatch.features.orb import ORBFeatureBackend
from lunarmatch.features.protocol import FeatureBackend
from lunarmatch.features.sift import SIFTFeatureBackend
from lunarmatch.models.config import FeaturesConfig

_BACKEND_REGISTRY: dict[str, Callable[[FeaturesConfig | None], FeatureBackend]] = {
    "orb": lambda cfg: ORBFeatureBackend(config=cfg),
    "sift": lambda cfg: SIFTFeatureBackend(config=cfg),
    "learned_lightglue": lambda cfg: LearnedMatcherAdapter(config=cfg),
    "learned_loftr": lambda cfg: LearnedLoFTRBackend(config=cfg),
}


def register_feature_backend(
    name: str, factory: Callable[[FeaturesConfig | None], FeatureBackend]
) -> None:
    """Register a new feature backend factory by name."""
    clean_name = name.lower().strip()
    if not clean_name:
        raise ValueError("Feature backend name cannot be empty")
    _BACKEND_REGISTRY[clean_name] = factory


def get_feature_backend(
    name: str | FeaturesConfig, config: FeaturesConfig | None = None
) -> FeatureBackend:
    """Get a feature backend instance by name or config object.

    Args:
        name: Backend name string (e.g. 'orb', 'sift') or FeaturesConfig instance.
        config: Optional FeaturesConfig instance if name is string.

    Returns:
        FeatureBackend instance.

    Raises:
        ValueError: If backend name is unknown.
    """
    cfg: FeaturesConfig | None
    if isinstance(name, FeaturesConfig):
        backend_name = name.backend.lower().strip()
        cfg = name
    else:
        backend_name = str(name).lower().strip()
        cfg = config

    if backend_name not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unknown feature backend '{backend_name}'. Available backends: {available}"
        )

    return _BACKEND_REGISTRY[backend_name](cfg)


def list_feature_backends() -> list[str]:
    """List registered feature backend names."""
    return list(_BACKEND_REGISTRY.keys())

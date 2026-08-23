"""LunarMatch package for multi-modal lunar image correspondence and registration."""

__version__ = "0.1.0"

from lunarmatch.orchestrator import RegistrationOrchestrator, register_pair

__all__ = [
    "RegistrationOrchestrator",
    "__version__",
    "register_pair",
]

from __future__ import annotations

import importlib
import platform

MODULES = ["numpy", "scipy", "cv2", "skimage", "yaml", "typer"]
print("Python:", platform.python_version())
failed = []
for name in MODULES:
    try:
        module = importlib.import_module(name)
        print(f"OK {name}: {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:  # noqa: BLE001
        failed.append((name, str(exc)))
        print(f"FAIL {name}: {exc}")
raise SystemExit(1 if failed else 0)

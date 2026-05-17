from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _get_cache_dir(override: Optional[str] = None) -> Path:
    """Return the root cache directory.

    Priority: override argument → RLDATA_CACHE env var → ~/.cache/rldata
    """
    if override is not None:
        return Path(override)
    env = os.environ.get("RLDATA_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "rldata"

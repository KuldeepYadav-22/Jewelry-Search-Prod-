# ──────────────────────────────────────────────────────────────────────────────
# BiRefNet is no longer loaded — background removal happens on-device.
# Uncomment if you need to re-enable server-side BG removal:
#
# from engines.birefnet_engine import BiRefNetEngine
# ──────────────────────────────────────────────────────────────────────────────

from engines.clip_engine import CLIPEngine
from engines.dinov2_engine import DINOv2Engine

__all__ = ["CLIPEngine", "DINOv2Engine"]

# To re-enable BiRefNet:
# __all__ = ["BiRefNetEngine", "CLIPEngine", "DINOv2Engine"]

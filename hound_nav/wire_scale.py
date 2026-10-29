"""Turn a body velocity into the twist the drive bridge should receive.

The bridge converts m/s using the speeds measured at the firmware's 0.35 ceiling
with the speed knob at maximum. The firmware then multiplies by that ceiling and
by the knob. The 0.35 is already in the measurement, so a velocity v is sent as
v / knob. The knob is the StamPLC's CH8 reading, 0..1, from the dashboard.
"""

import json
import time
import urllib.request

_URL = "http://127.0.0.1:8080/api/stamplc"
_cache_t = 0.0
_cache_k = 1.0


def knob() -> float:
    """CH8 as 0..1. Stays at 1 when the dashboard has no reading, so a missed poll does not amplify."""
    global _cache_t, _cache_k
    now = time.monotonic()
    if now - _cache_t < 0.5:
        return _cache_k
    try:
        with urllib.request.urlopen(_URL, timeout=0.3) as resp:
            spd = json.load(resp)["drive"]["spd"]
        if spd is not None and float(spd) > 0.0:
            _cache_k = min(1.0, float(spd))
    except (OSError, KeyError, TypeError, ValueError):
        pass
    _cache_t = now
    return _cache_k


def to_wire(vx: float, vy: float, wz: float) -> tuple[float, float, float]:
    k = knob()
    return vx / k, vy / k, wz / k

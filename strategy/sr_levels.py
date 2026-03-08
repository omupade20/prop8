from typing import List, Dict, Optional, Tuple
from statistics import mean


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _find_local_extrema(values: List[float], window: int = 5) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:

    n = len(values)
    maxima, minima = [], []

    if n < window * 2 + 1:
        return maxima, minima

    half = window // 2

    for i in range(half, n - half):

        center = values[i]
        left = values[i - half:i]
        right = values[i + 1:i + 1 + half]

        if all(center > x for x in left + right):
            maxima.append((i, center))

        if all(center < x for x in left + right):
            minima.append((i, center))

    return maxima, minima


def _cluster_levels(peaks: List[float], tol_pct: float = 0.004) -> List[Dict]:

    if not peaks:
        return []

    sorted_peaks = sorted(peaks)

    clusters = []
    cluster = [sorted_peaks[0]]

    for p in sorted_peaks[1:]:

        avg = sum(cluster) / len(cluster)
        tol = avg * tol_pct

        if abs(p - avg) <= tol:
            cluster.append(p)
        else:
            clusters.append(cluster)
            cluster = [p]

    clusters.append(cluster)

    out = []

    for c in clusters:

        lvl = mean(c)

        out.append({
            "level": round(lvl, 6),
            "count": len(c),
            "strength": min(len(c), 4)
        })

    return out


# -------------------------------------------------
# MAIN SR CALCULATION (5m candles)
# -------------------------------------------------

def compute_sr_levels_from_5m(
    candles_5m: List[Dict],
    lookback: int = 120,
    extrema_window: int = 5,
    cluster_tol_pct: float = 0.004,
    max_levels: int = 3
) -> Dict[str, List[Dict]]:

    if not candles_5m:
        return {"supports": [], "resistances": []}

    candles = candles_5m[-lookback:]

    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    max_extrema, _ = _find_local_extrema(highs, window=extrema_window)
    _, min_extrema = _find_local_extrema(lows, window=extrema_window)

    resistances = [val for _, val in max_extrema]
    supports = [val for _, val in min_extrema]

    resist_clusters = _cluster_levels(resistances, tol_pct=cluster_tol_pct)
    supp_clusters = _cluster_levels(supports, tol_pct=cluster_tol_pct)

    supp_sorted = sorted(supp_clusters, key=lambda x: x["level"])[:max_levels]
    res_sorted = sorted(resist_clusters, key=lambda x: x["level"], reverse=True)[:max_levels]

    return {
        "supports": supp_sorted,
        "resistances": res_sorted
    }


# -------------------------------------------------
# BACKWARD COMPATIBILITY (OLD SYSTEM SUPPORT)
# -------------------------------------------------

def compute_sr_levels(
    highs: List[float],
    lows: List[float],
    lookback: int = 120
) -> Dict[str, List[Dict]]:
    """
    Compatibility wrapper so older modules don't break.
    Converts highs/lows into pseudo candles.
    """

    if not highs or not lows:
        return {"supports": [], "resistances": []}

    highs = highs[-lookback:]
    lows = lows[-lookback:]

    candles = []

    for h, l in zip(highs, lows):
        candles.append({
            "high": h,
            "low": l
        })

    return compute_sr_levels_from_5m(candles)


# -------------------------------------------------
# NEAREST SR
# -------------------------------------------------

def get_nearest_sr(
    price: float,
    sr_levels: Dict[str, List[Dict]],
    max_search_pct: float = 0.03
) -> Optional[Dict]:

    supports = sr_levels.get("supports", [])
    resistances = sr_levels.get("resistances", [])

    best = None
    best_dist = float("inf")

    for s in supports:

        lvl = s["level"]
        dist = abs(price - lvl) / max(lvl, 1e-9)

        if dist < best_dist:
            best_dist = dist
            best = {
                "type": "support",
                "level": lvl,
                "dist_pct": dist,
                "strength": s.get("strength", 1)
            }

    for r in resistances:

        lvl = r["level"]
        dist = abs(lvl - price) / max(price, 1e-9)

        if dist < best_dist:
            best_dist = dist
            best = {
                "type": "resistance",
                "level": lvl,
                "dist_pct": dist,
                "strength": r.get("strength", 1)
            }

    if best and best["dist_pct"] <= max_search_pct:
        return best

    return None


# -------------------------------------------------
# LOCATION SCORE
# -------------------------------------------------

def sr_location_score(
    price: float,
    nearest_sr: Optional[Dict],
    direction: str,
    proximity_threshold: float = 0.02
) -> float:

    if nearest_sr is None:
        return 0.0

    dist = nearest_sr.get("dist_pct")

    if dist is None or dist > proximity_threshold:
        return 0.0

    closeness = (proximity_threshold - dist) / proximity_threshold

    strength = nearest_sr.get("strength", 1)
    strength_factor = min(1.5, 0.6 + 0.2 * strength)

    typ = nearest_sr["type"]

    if direction == "LONG":
        sign = 1 if typ == "support" else -1
    elif direction == "SHORT":
        sign = 1 if typ == "resistance" else -1
    else:
        sign = 0

    score = sign * closeness * strength_factor

    return round(max(min(score, 1.0), -1.0), 3)

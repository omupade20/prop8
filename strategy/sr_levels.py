# strategy/sr_levels.py
"""
Support & Resistance utilities (lightweight, deterministic, intraday-ready).

Functions:
- compute_simple_sr: legacy min/max over lookback
- compute_sr_levels: find local highs/lows, cluster them into SR levels with strengths
- get_nearest_sr: nearest support/resistance to current price (with distance)
- sr_distance: distance metrics (pct)
- sr_location_score: soft score (-1..+1) that rewards/penalizes a direction based on proximity to SR

Design principles:
- Conservative: S/R are used as soft inputs (not hard veto).
- Memory-safe: works with basic lists of highs/lows.
- Configurable proximity thresholds for clustering and location scoring.
"""

from typing import List, Dict, Optional, Tuple
from statistics import mean


def compute_simple_sr(highs: List[float], lows: List[float], lookback: int = 120) -> Dict[str, float]:
    """
    Legacy simple S/R: max(highs) and min(lows) over lookback.
    """
    highs = highs[-lookback:] if highs else []
    lows = lows[-lookback:] if lows else []

    if not highs or not lows:
        return {"support": None, "resistance": None}

    return {
        "support": min(lows),
        "resistance": max(highs)
    }


def _find_local_extrema(values: List[float], window: int = 5) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Find approximate local maxima (resistances) and minima (supports).
    Returns two lists of (index, value).
    Window should be small (e.g., 3-7). Edges are ignored.
    """
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


def _cluster_levels(peaks: List[float], tol_pct: float = 0.005) -> List[Dict]:
    """
    Cluster numeric peaks into levels using a simple linear scan.
    tol_pct: cluster tolerance relative to the price (e.g., 0.005 = 0.5%)
    Returns list of {level, count, strength} sorted by level ascending.
    Strength ~ count (number of peaks in cluster).
    """
    if not peaks:
        return []

    sorted_peaks = sorted(peaks)
    clusters = []
    cluster = [sorted_peaks[0]]

    for p in sorted_peaks[1:]:
        ref = cluster[-1]
        # use average of cluster as reference to be stable
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
        out.append({"level": round(lvl, 6), "count": len(c), "strength": len(c)})  # strength == count for now

    return out


def compute_sr_levels(
    highs: List[float],
    lows: List[float],
    lookback: int = 240,
    extrema_window: int = 5,
    cluster_tol_pct: float = 0.005,
    max_levels: int = 5
) -> Dict[str, List[Dict]]:
    """
    Compute clustered support & resistance levels from recent highs/lows.
    - lookback: how many bars to analyze (1-min bars)
    - extrema_window: window for local extrema detection (3-7 recommended)
    - cluster_tol_pct: clustering tolerance as fraction of price (e.g., 0.005 -> 0.5%)
    - max_levels: cap number of levels returned per side
    Returns:
      {"supports": [...], "resistances": [...]}
    Each item: {"level": price, "count": n, "strength": n}
    """
    highs_s = highs[-lookback:] if highs else []
    lows_s = lows[-lookback:] if lows else []

    if not highs_s or not lows_s:
        return {"supports": [], "resistances": []}

    # find local peaks/valleys
    res_peaks, sup_peaks = _find_local_extrema(highs_s, window=extrema_window)
    # _find_local_extrema uses the same list for both; but we want maxima from highs_s and minima from lows_s
    # compute maxima from highs_s and minima from lows_s explicitly
    max_extrema, _ = _find_local_extrema(highs_s, window=extrema_window)
    _, min_extrema = _find_local_extrema(lows_s, window=extrema_window)

    resistances = [val for _, val in max_extrema]
    supports = [val for _, val in min_extrema]

    resist_clusters = _cluster_levels(resistances, tol_pct=cluster_tol_pct)
    supp_clusters = _cluster_levels(supports, tol_pct=cluster_tol_pct)

    # sort by price: supports ascending, resistances descending (useful)
    supp_clusters_sorted = sorted(supp_clusters, key=lambda x: x["level"])[:max_levels]
    res_clusters_sorted = sorted(resist_clusters, key=lambda x: x["level"], reverse=True)[:max_levels]

    return {"supports": supp_clusters_sorted, "resistances": res_clusters_sorted}


def get_nearest_sr(
    price: float,
    sr_levels: Dict[str, List[Dict]],
    max_search_pct: float = 0.03
) -> Optional[Dict]:
    """
    Find nearest support OR resistance within max_search_pct of price.
    Returns dict:
      {
        "type": "support" | "resistance",
        "level": float,
        "dist_pct": float (positive, e.g., 0.01 means 1%),
        "strength": int
      }
    If none found within threshold, returns None.
    """
    if not sr_levels:
        return None

    supports = sr_levels.get("supports", [])
    resistances = sr_levels.get("resistances", [])

    best = None
    best_dist = float("inf")

    for s in supports:
        lvl = s["level"]
        dist = abs(price - lvl) / max(lvl, 1e-9)
        if dist < best_dist:
            best_dist = dist
            best = {"type": "support", "level": lvl, "dist_pct": dist, "strength": s.get("strength", 1)}

    for r in resistances:
        lvl = r["level"]
        dist = abs(lvl - price) / max(price, 1e-9)
        if dist < best_dist:
            best_dist = dist
            best = {"type": "resistance", "level": lvl, "dist_pct": dist, "strength": r.get("strength", 1)}

    if best and best["dist_pct"] <= max_search_pct:
        return best
    return None


def sr_distance(price: float, sr: Dict) -> Optional[Dict[str, float]]:
    """
    Returns distance to support/resistance in fractional form.
    sr: {"support":..., "resistance":...} OR nearest_sr dict from get_nearest_sr
    """
    if not sr:
        return None

    if "support" in sr and "resistance" in sr:
        sup = sr.get("support")
        res = sr.get("resistance")
        if sup is None or res is None:
            return None
        return {
            "dist_to_support": (price - sup) / max(sup, 1e-9),
            "dist_to_resistance": (res - price) / max(price, 1e-9)
        }

    # if nearest_sr style
    if sr.get("type") == "support":
        return {"dist_to_support": (price - sr["level"]) / max(sr["level"], 1e-9)}
    if sr.get("type") == "resistance":
        return {"dist_to_resistance": (sr["level"] - price) / max(price, 1e-9)}
    return None


def sr_location_score(
    price: float,
    nearest_sr: Optional[Dict],
    direction: str,
    proximity_threshold: float = 0.015
) -> float:
    """
    Soft score in range [-1.0 .. +1.0] indicating how favorable current location is for `direction`.
    - direction: "LONG" or "SHORT"
    - nearest_sr: result from get_nearest_sr()
    - proximity_threshold: distance (fraction) below which influence is maximal (e.g., 0.03 = 3%)

    Scoring logic:
    - If LONG and nearest is support and within proximity_threshold -> positive (0..+1) scaled by closeness & strength.
    - If LONG and nearest is resistance within threshold -> negative (0..-1).
    - If SHORT and nearest is resistance -> positive; if SHORT and nearest is support -> negative.
    - If no nearest_sr or outside threshold -> 0 (neutral).
    """
    if nearest_sr is None:
        return 0.0

    dist = nearest_sr.get("dist_pct", None)
    if dist is None:
        return 0.0

    # if outside influence zone -> neutral
    if dist > proximity_threshold:
        return 0.0

    # closeness factor: 1.0 at dist=0, 0 at dist=proximity_threshold
    closeness = max(0.0, (proximity_threshold - dist) / (proximity_threshold + 1e-9))

    # strength modifier based on how many peaks formed this level
    strength = float(nearest_sr.get("strength", 1))
    strength_factor = min(1.5, 0.5 + 0.25 * strength)  # 1..1.5 approx

    sign = 0
    typ = nearest_sr.get("type")
    if direction == "LONG":
        if typ == "support":
            sign = 1
        elif typ == "resistance":
            sign = -1
    elif direction == "SHORT":
        if typ == "resistance":
            sign = 1
        elif typ == "support":
            sign = -1

    score = sign * closeness * strength_factor
    # clamp
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return round(score, 3)

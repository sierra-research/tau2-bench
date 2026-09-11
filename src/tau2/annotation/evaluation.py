# Copyright Sierra
"""Instrument-trust primitives shared by the calibration reports.

**Chance-corrected agreement** (Cohen's kappa) over 2x2 or KxK confusion
matrices — raw percent agreement flatters any instrument whose classes are
imbalanced.

**The LIVE/shadow gate**: a judge axis gets a vote only when its adjudicated
precision clears the bar; below it the axis still runs and is still reported,
just loudly provisional. Judge calibration is measured per dimension and per
language, never pooled — a judge that correlates 0.7 overall and 0.4 on the
one language whose calls are worst is not a judge that works, and pooling
hides exactly the failure that matters.
"""

from typing import Literal, Optional

import numpy as np

from tau2.config import DEFAULT_ANNOTATION_PRECISION_BAR

Gate = Literal["LIVE", "shadow"]


def gate(precision: Optional[float], precision_bar: float = None) -> Gate:
    """LIVE when adjudicated precision clears the bar, shadow otherwise.

    Shadow means the axis still runs and is still reported — it just does not
    get a vote. An axis nobody can trust that silently keeps scoring is worse
    than one that is loudly provisional.
    """
    bar = DEFAULT_ANNOTATION_PRECISION_BAR if precision_bar is None else precision_bar
    return "LIVE" if precision is not None and precision >= bar else "shadow"


def cohens_kappa_from_matrix(counts: np.ndarray) -> Optional[float]:
    """Chance-corrected agreement over a KxK confusion matrix.

    General in K because agreement tasks routinely have more than two answers
    (e.g. A, B, and "I could not separate them"). Collapsing the third class
    into one of the sides scores a rater who called it even and one who picked
    a clear winner as agreeing.
    """
    total = float(counts.sum())
    if total == 0:
        return None
    observed = float(np.trace(counts)) / total
    rows = counts.sum(axis=1) / total
    cols = counts.sum(axis=0) / total
    expected = float(rows @ cols)
    if expected == 1.0:
        # Both raters constant, which can only happen when they agree on every
        # item. Kappa is 0/0 there; the defined convention is perfect
        # agreement, and returning None would read as "no data".
        return 1.0
    return (observed - expected) / (1.0 - expected)


def cohens_kappa(tp: int, fn: int, fp: int, tn: int) -> Optional[float]:
    """Chance-corrected agreement over a 2x2 confusion matrix."""
    return cohens_kappa_from_matrix(np.array([[tp, fn], [fp, tn]], dtype=float))

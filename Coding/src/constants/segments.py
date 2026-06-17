# segments.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentConfig:
    """Segment-specific modelling parameters, tunable independently per group."""
    churn_threshold_days: int

PERSONAL = SegmentConfig(churn_threshold_days=160)

PROFESSIONAL = SegmentConfig(churn_threshold_days=80)
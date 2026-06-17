# Personal | Professional split
from dataclasses import dataclass

@dataclass(frozen=True)
class SegmentConfig:
    hhi_threshold: int
    car_share_abs: int
    car_share_fraction: float
    quantile_filter: float
    burst_time_hr: float

PERSONAL = SegmentConfig(
    hhi_threshold=6,
    car_share_abs=4,
    car_share_fraction=0.8,
    quantile_filter=0.9,
    burst_time_hr=1/30
)

PROFESSIONAL = SegmentConfig(
    hhi_threshold=6,
    car_share_abs=4,
    car_share_fraction=0.8,
    quantile_filter=0.9,
    burst_time_hr=1/30
)
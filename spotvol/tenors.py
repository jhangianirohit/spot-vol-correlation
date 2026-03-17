DAYS_PER_YEAR = 365

# Calendar days to expiry
TENOR_DAYS = {
    "ON": 1, "1D": 1,
    "1W": 7, "2W": 14, "3W": 21,
    "1M": 30, "2M": 60,
    "3M": 91, "6M": 182,
    "9M": 274, "1Y": 365,
    "2Y": 730,
}

# Canonical ordering for display
TENOR_ORDER = ["ON", "1D", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y"]


def tenor_to_T(label: str) -> float:
    """Convert tenor label to time in years (calendar days / 365)."""
    return TENOR_DAYS[label.upper()] / DAYS_PER_YEAR


def tenor_to_ndays(label: str) -> int:
    """Convert tenor label to number of calendar days."""
    return TENOR_DAYS[label.upper()]


def sort_tenors(labels: list[str]) -> list[str]:
    """Sort tenor labels in chronological order."""
    order = {t: i for i, t in enumerate(TENOR_ORDER)}
    return sorted(labels, key=lambda x: order.get(x.upper(), 999))

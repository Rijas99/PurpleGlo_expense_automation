from __future__ import annotations

from datetime import datetime


MONTHS = [
    ("Jan", "January"),
    ("Feb", "February"),
    ("Mar", "March"),
    ("Apr", "April"),
    ("May", "May"),
    ("Jun", "June"),
    ("Jul", "July"),
    ("Aug", "August"),
    ("Sep", "September"),
    ("Oct", "October"),
    ("Nov", "November"),
    ("Dec", "December"),
]
MONTH_ABBRS = [abbr for abbr, _ in MONTHS]


def current_month_label(now: datetime | None = None) -> str:
    moment = now or datetime.now()
    return f"{MONTH_ABBRS[moment.month - 1]} {moment.year}"


def previous_month_label(now: datetime | None = None) -> str:
    moment = now or datetime.now()
    if moment.month == 1:
        return f"Dec {moment.year - 1}"
    return f"{MONTH_ABBRS[moment.month - 2]} {moment.year}"


def month_slug(label: str) -> str:
    return label.replace(" ", "_")


def current_month_slug(now: datetime | None = None) -> str:
    return month_slug(current_month_label(now))


def previous_month_slug(now: datetime | None = None) -> str:
    return month_slug(previous_month_label(now))


def recent_month_options(count: int = 12, now: datetime | None = None) -> list[tuple[str, str]]:
    moment = now or datetime.now()
    year, month = moment.year, moment.month
    options: list[tuple[str, str]] = []
    for _ in range(count):
        label = f"{MONTH_ABBRS[month - 1]} {year}"
        options.append((label, month_slug(label)))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return options


def month_slug_from_display_date(value: str, now: datetime | None = None) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    moment = now or datetime.now()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d-%b", "%d-%B"):
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        year = parsed.year if "%Y" in fmt else moment.year
        month = parsed.month
        if "%Y" not in fmt and month > moment.month:
            year -= 1
        return f"{MONTH_ABBRS[month - 1]}_{year}"
    return None

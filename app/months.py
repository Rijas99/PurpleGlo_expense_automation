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

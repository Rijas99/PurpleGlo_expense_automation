from __future__ import annotations

import copy
import re
from datetime import datetime, time as dt_time

from app.constants import CATEGORIES, FOOD_CAP_AMOUNT

_AMOUNT_RE = re.compile(
    r"^\s*(?:amount|amt|price|total)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*$",
    re.I,
)
_DATE_RE = re.compile(
    r"^\s*date\s*[:=]?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\w{3}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*$",
    re.I,
)
_DESC_RE = re.compile(r"^\s*(?:description|desc)\s*[:=]?\s*(.+)\s*$", re.I)
_PROJECT_NAME_RE = re.compile(r"^\s*project\s*name\s*[:=]?\s*(.+)\s*$", re.I)
_PROJECT_RE = re.compile(r"^\s*project(?:\s*code)?\s*[:=]?\s*(.+)\s*$", re.I)
_CATEGORY_RE = re.compile(r"^\s*(?:category|cat)\s*[:=]?\s*(.+)\s*$", re.I)
_CAP_RE = re.compile(r"^\s*cap(?:\s*(?:to\s*)?40)?\s*$", re.I)


def get_meal_description(time_obj) -> str:
    if isinstance(time_obj, datetime):
        time_obj = time_obj.time()
    elif isinstance(time_obj, str):
        try:
            hour, minute = map(int, time_obj.split(":")[:2])
            time_obj = dt_time(hour, minute)
        except ValueError:
            time_obj = datetime.now().time()
    elif not isinstance(time_obj, dt_time):
        time_obj = datetime.now().time()

    hour = time_obj.hour
    if 5 <= hour < 12:
        return "Breakfast"
    if 12 <= hour < 18:
        return "Lunch"
    return "Dinner"


def _match_category(text: str) -> str | None:
    needle = text.strip().lower()
    for cat in CATEGORIES:
        if cat.lower() == needle:
            return cat
    for cat in CATEGORIES:
        if needle in cat.lower() or cat.lower() in needle:
            return cat
    return None


def apply_corrections(draft: dict, text: str) -> dict:
    """Return a new draft with fields updated from a user reply. Unknown lines are ignored."""
    updated = copy.deepcopy(draft)
    if not text or not text.strip():
        return updated

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _AMOUNT_RE.match(line)
        if m:
            updated["amount"] = float(m.group(1))
            continue

        m = _DATE_RE.match(line)
        if m:
            updated["date"] = _normalize_date(m.group(1))
            continue

        m = _DESC_RE.match(line)
        if m:
            updated["description"] = m.group(1).strip()
            continue

        m = _PROJECT_NAME_RE.match(line)
        if m:
            updated["project_name"] = m.group(1).strip()
            continue

        m = _CATEGORY_RE.match(line)
        if m:
            matched = _match_category(m.group(1))
            if matched:
                updated["category"] = matched
            continue

        m = _PROJECT_RE.match(line)
        if m:
            updated["project_code"] = m.group(1).strip()
            continue

        if _CAP_RE.match(line):
            _apply_food_cap(updated)
            continue

        matched = _match_category(line)
        if matched and line.lower() == matched.lower():
            updated["category"] = matched
            continue

        updated["project_name"] = line

    return updated


def _apply_food_cap(draft: dict) -> None:
    category = str(draft.get("category") or "")
    try:
        amount = float(draft.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0.0
    if category == "Food & Beverages" and amount > FOOD_CAP_AMOUNT:
        desc = str(draft.get("description") or "").strip()
        if "capped at 40" not in desc:
            draft["description"] = f"{desc} (capped at 40)".strip()
        draft["amount"] = FOOD_CAP_AMOUNT


def _normalize_date(raw: str) -> str:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%d-%b", "%d/%b", "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y"):
        try:
            parsed = datetime.strptime(raw, fmt)
            if fmt in ("%d-%b", "%d/%b"):
                return parsed.strftime("%Y-%m-%d")
            if "%Y" not in fmt and "%y" not in fmt:
                return parsed.replace(year=datetime.now().year).strftime("%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def format_display_date(iso_or_display: str) -> str:
    """Expense form date column uses d-mmm like 15-Aug."""
    if not iso_or_display:
        return datetime.now().strftime("%d-%b")
    for fmt in ("%Y-%m-%d", "%d-%b", "%d-%B"):
        try:
            return datetime.strptime(iso_or_display, fmt).strftime("%d-%b")
        except ValueError:
            continue
    return iso_or_display


def to_input_date(display: str, year: int | None = None) -> str:
    """Turn 15-Aug / ISO into an HTML date input value."""
    if not display:
        return datetime.now().strftime("%Y-%m-%d")
    year = year or datetime.now().year
    raw = display.strip()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        pass
    for fmt in ("%d-%b-%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(f"{raw}-{year}", fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return datetime.now().strftime("%Y-%m-%d")

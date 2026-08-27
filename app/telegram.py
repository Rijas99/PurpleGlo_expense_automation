from __future__ import annotations

import json
import re

from app.corrections import format_display_date


def format_draft_message(draft: dict) -> str:
    date = draft.get("date") or "—"
    try:
        date_show = format_display_date(str(date))
    except Exception:
        date_show = date
    amount = draft.get("amount")
    try:
        amount_show = f"{float(amount):.2f}"
    except (TypeError, ValueError):
        amount_show = str(amount or "—")

    kind = draft.get("kind") or "receipt"
    missing = []
    if not str(draft.get("project_name") or "").strip():
        missing.append("project name")
    if kind == "transport":
        if not str(draft.get("from_location") or "").strip():
            missing.append("from")
        if not str(draft.get("destination") or "").strip():
            missing.append("destination")

    title = {
        "receipt": "Receipt draft",
        "credit_card": "Credit card draft",
        "transport": "Transport draft",
    }.get(kind, "Draft")

    lines = [title, f"Date: {date_show}"]
    if kind == "transport":
        lines.extend(
            [
                f"From: {draft.get('from_location') or '—'}",
                f"Destination: {draft.get('destination') or '—'}",
                f"Return: {'yes' if draft.get('return_included') else 'no'}",
                f"Project name: {draft.get('project_name') or '—'}",
            ]
        )
    else:
        lines.extend(
            [
                f"Time: {draft.get('time') or '—'}",
                f"Description: {draft.get('description') or '—'}",
                f"Category: {draft.get('category') or '—'}",
                f"Amount: {amount_show}",
                f"Project name: {draft.get('project_name') or '—'}",
            ]
        )
    lines.extend(
        [
            "",
            "Caption shortcuts:",
            "photo + adnoc  →  receipt, project name adnoc",
            "photo + CC, adnoc  →  credit card",
            "TR, Dubai, Abu Dhabi, adnoc  →  transport",
            "",
            "Send save / ok when it looks right.",
        ]
    )
    if missing:
        lines.append("")
        lines.append("Still needed before save: " + ", ".join(missing))
    return "\n".join(lines)


def parse_telegram_note(text: str) -> dict:
    """Parse a photo caption or text command.

    Receipts: the whole caption is the project name (e.g. adnoc).
    Credit card: CC, project name
    Transport: TR, from, destination, project name [, return]
    """
    raw = (text or "").strip()
    if not raw:
        return {"kind": "receipt", "project_name": ""}

    cc = re.match(r"^\s*cc\s*[,:]\s*(.+)\s*$", raw, re.I)
    if cc:
        return {"kind": "credit_card", "project_name": cc.group(1).strip()}

    tr = re.match(r"^\s*(?:tr|transport)\s*[,:]\s*(.+)\s*$", raw, re.I)
    if tr:
        parts = [p.strip() for p in tr.group(1).split(",") if p.strip()]
        return_included = False
        if parts and parts[-1].lower() in {"return", "return included", "yes", "round trip"}:
            return_included = True
            parts = parts[:-1]
        return {
            "kind": "transport",
            "from_location": parts[0] if len(parts) > 0 else "",
            "destination": parts[1] if len(parts) > 1 else "",
            "project_name": ", ".join(parts[2:]) if len(parts) > 2 else "",
            "return_included": return_included,
        }

    return {"kind": "receipt", "project_name": raw}


def parse_save_command(text: str) -> bool:
    if not text:
        return False
    return text.strip().lower() in {"save", "ok", "okay", "yes", "confirm", "✅"}


def is_start_command(text: str) -> bool:
    if not text:
        return False
    lowered = text.strip().lower()
    return lowered in {"/start", "start", "help", "/help"} or lowered.startswith("/start")


def dumps_draft(draft: dict) -> str:
    safe = {k: v for k, v in draft.items() if k != "image_bytes"}
    return json.dumps(safe)


def loads_draft(payload: str) -> dict:
    return json.loads(payload)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")

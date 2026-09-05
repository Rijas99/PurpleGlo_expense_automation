from __future__ import annotations

import json
import re

from app.corrections import format_display_date


_DIS_INLINE_RE = re.compile(r"\bdis\s*:\s*(.+)$", re.I)
_CAP_INLINE_RE = re.compile(r"\bcap\s*[,:]\s*(\d+(?:\.\d+)?)\s*$", re.I)
_CAP_LINE_RE = re.compile(r"^\s*cap(?:\s*(?:to)?\s*[,:]?\s*(\d+(?:\.\d+)?))?\s*$", re.I)
_DIS_LINE_RE = re.compile(r"^\s*dis\s*:\s*(.+)\s*$", re.I)


def _split_caption(text: str) -> tuple[str, str, float | None]:
    extra_bits: list[str] = []
    cap_amount: float | None = None
    kept: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cap_line = _CAP_LINE_RE.match(line)
        if cap_line:
            cap_amount = float(cap_line.group(1)) if cap_line.group(1) else 40.0
            continue
        dis_line = _DIS_LINE_RE.match(line)
        if dis_line:
            extra_bits.append(dis_line.group(1).strip())
            continue
        cap_inline = _CAP_INLINE_RE.search(line)
        if cap_inline:
            cap_amount = float(cap_inline.group(1))
            line = line[: cap_inline.start()].strip(" ,")
        dis_inline = _DIS_INLINE_RE.search(line)
        if dis_inline:
            extra_bits.append(dis_inline.group(1).strip())
            line = line[: dis_inline.start()].strip(" ,")
        if line:
            kept.append(line)
    extra = ", ".join(part for part in extra_bits if part)
    return "\n".join(kept), extra, cap_amount


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
            "photo + adnoc / dis: ali and rijas  →  Lunch, ali and rijas",
            "photo + cap,40  →  cap amount to 40",
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

    Receipts: project name, optional dis: extra text, optional cap,40.
    Credit card: CC, project name
    Transport: TR, from, destination, project name [, return]
    """
    raw = (text or "").strip()
    if not raw:
        return {"kind": "receipt", "project_name": "", "extra_description": "", "cap_amount": None}

    remainder, extra, cap_amount = _split_caption(raw)
    remainder = remainder.strip()
    parsed: dict
    if not remainder:
        parsed = {"kind": "receipt", "project_name": ""}
    else:
        cc = re.match(r"^\s*cc\s*[,:]\s*(.+)\s*$", remainder, re.I)
        if cc:
            parsed = {"kind": "credit_card", "project_name": cc.group(1).strip()}
        else:
            tr = re.match(r"^\s*(?:tr|transport)\s*[,:]\s*(.+)\s*$", remainder, re.I)
            if tr:
                parts = [p.strip() for p in tr.group(1).split(",") if p.strip()]
                return_included = False
                if parts and parts[-1].lower() in {"return", "return included", "yes", "round trip"}:
                    return_included = True
                    parts = parts[:-1]
                parsed = {
                    "kind": "transport",
                    "from_location": parts[0] if len(parts) > 0 else "",
                    "destination": parts[1] if len(parts) > 1 else "",
                    "project_name": ", ".join(parts[2:]) if len(parts) > 2 else "",
                    "return_included": return_included,
                }
            else:
                parsed = {"kind": "receipt", "project_name": remainder}

    parsed["extra_description"] = extra
    parsed["cap_amount"] = cap_amount
    return parsed


def parse_save_command(text: str) -> bool:
    if not text:
        return False
    return text.strip().lower() in {"save", "ok", "okay", "yes", "confirm", "✅"}


_HELP_WORDS = {
    "start",
    "help",
    "commands",
    "menu",
    "hi",
    "hii",
    "hiii",
    "hy",
    "hye",
    "hey",
    "heyy",
    "hello",
    "helloo",
    "helo",
    "hlo",
    "hlw",
    "hai",
    "yo",
    "hola",
    "howdy",
    "morning",
    "evening",
    "sup",
    "wassup",
    "salam",
    "salaam",
}
_HELP_PHRASES = {
    "hi there",
    "hey there",
    "hello there",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "whats up",
    "what's up",
}


def is_start_command(text: str) -> bool:
    if not text:
        return False
    lowered = text.strip().lower()
    if lowered.startswith(("/start", "/help", "/commands", "/menu")):
        return True
    cleaned = re.sub(r"[!.?]+$", "", lowered).strip()
    return cleaned in _HELP_WORDS or cleaned in _HELP_PHRASES


def commands_help_text() -> str:
    return (
        "How to add expenses:\n\n"
        "Receipts — send a photo. Caption = project name, optional extra lines:\n"
        "adnoc\n"
        "dis: ali and rijas\n"
        "cap,40\n\n"
        "Food bills become Breakfast / Lunch / Dinner from the time on the receipt.\n"
        "Breakfast 5:00–11:15, Lunch 11:15–18:15, Dinner after that.\n\n"
        "Credit card — send a photo, caption:\n"
        "CC, adnoc\n\n"
        "Transport — text only, no photo:\n"
        "TR, Dubai, Abu Dhabi, adnoc\n"
        "TR, Dubai, Abu Dhabi, adnoc, return\n\n"
        "Then tap Save or send save."
    )


def dumps_draft(draft: dict) -> str:
    safe = {k: v for k, v in draft.items() if k != "image_bytes"}
    return json.dumps(safe)


def loads_draft(payload: str) -> dict:
    return json.loads(payload)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")

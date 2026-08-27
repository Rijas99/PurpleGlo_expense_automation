from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.config import settings
from app.corrections import apply_corrections, format_display_date, get_meal_description
from app.db import (
    add_credit_card,
    add_receipt,
    add_transport,
    delete_draft,
    get_draft,
    next_receipt_ref,
    upsert_draft,
)
from app.gemini import analyze_receipt_bytes
from app.telegram import (
    dumps_draft,
    format_draft_message,
    is_start_command,
    loads_draft,
    parse_save_command,
    parse_telegram_note,
)

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
API = "https://api.telegram.org"


class TelegramError(Exception):
    pass


def _api(method: str, **kwargs) -> dict:
    token = settings.telegram_bot_token
    if not token:
        raise TelegramError("TELEGRAM_BOT_TOKEN is not set.")
    url = f"{API}/bot{token}/{method}"
    with httpx.Client(timeout=60.0) as client:
        if "files" in kwargs:
            files = kwargs.pop("files")
            resp = client.post(url, data=kwargs, files=files)
        else:
            resp = client.post(url, json=kwargs)
        data = resp.json()
    if not data.get("ok"):
        raise TelegramError(str(data))
    return data["result"]


def send_message(chat_id: int | str, text: str, reply_markup: dict | None = None) -> None:
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    _api("sendMessage", **payload)


def save_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [{"text": "Save receipt", "callback_data": "save"}],
            [{"text": "Cancel draft", "callback_data": "cancel"}],
        ]
    }


def delete_webhook() -> None:
    _api("deleteWebhook", drop_pending_updates=False)


def image_file_id(message: dict) -> str | None:
    photos = message.get("photo") or []
    if photos:
        return photos[-1]["file_id"]
    document = message.get("document") or {}
    mime = str(document.get("mime_type") or "").lower()
    name = str(document.get("file_name") or "").lower()
    if document.get("file_id") and (
        mime.startswith("image/")
        or name.endswith((".jpg", ".jpeg", ".png", ".webp", ".heic"))
    ):
        return document["file_id"]
    return None


def download_file(file_id: str) -> bytes:
    info = _api("getFile", file_id=file_id)
    file_path = info["file_path"]
    url = f"{API}/file/bot{settings.telegram_bot_token}/{file_path}"
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def user_allowed(user_id: int | None) -> bool:
    allowed = settings.allowed_telegram_ids()
    if not allowed:
        return True
    return str(user_id or "") in allowed


def _finalize_description(draft: dict) -> dict:
    category = draft.get("category")
    if category == "Food & Beverages" and not str(draft.get("description") or "").strip():
        draft["description"] = get_meal_description(draft.get("time"))
    return draft


def build_receipt_draft(extracted: dict, caption: str = "") -> dict:
    note = parse_telegram_note(caption)
    draft = {
        "kind": note.get("kind") or "receipt",
        "date": extracted.get("date") or "",
        "time": extracted.get("time"),
        "amount": extracted.get("amount") or 0,
        "description": extracted.get("description") or "",
        "category": extracted.get("category") or "Can't classify",
        "project_code": "",
        "project_name": note.get("project_name") or "",
        "from_location": note.get("from_location") or "",
        "destination": note.get("destination") or "",
        "return_included": bool(note.get("return_included")),
    }
    return _finalize_description(draft)


def build_transport_draft(note: dict) -> dict:
    from datetime import datetime

    return {
        "kind": "transport",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "from_location": note.get("from_location") or "",
        "destination": note.get("destination") or "",
        "return_included": bool(note.get("return_included")),
        "project_code": "",
        "project_name": note.get("project_name") or "",
        "amount": 0,
        "description": "",
        "category": "Transportation",
    }


def _save_if_ready(db_path, chat_id: str, draft: dict, image_bytes: bytes | None, image_mime: str) -> str:
    project_name = str(draft.get("project_name") or "").strip()
    if not project_name:
        return "Project name is required. Send the photo again with the project name as the caption, e.g. adnoc"
    kind = draft.get("kind") or "receipt"
    date = format_display_date(str(draft.get("date") or ""))

    if kind == "transport":
        from_location = str(draft.get("from_location") or "").strip()
        destination = str(draft.get("destination") or "").strip()
        if not from_location or not destination:
            return "Transport needs: TR, from, destination, project name"
        add_transport(
            db_path,
            {
                "date": date,
                "from_location": from_location,
                "destination": destination,
                "return_included": bool(draft.get("return_included")),
                "project_code": str(draft.get("project_code") or "").strip(),
                "project_name": project_name,
            },
        )
        delete_draft(db_path, chat_id)
        return f"Saved transport {from_location} → {destination} ({project_name})"

    if kind == "credit_card":
        add_credit_card(
            db_path,
            {
                "date": date,
                "description": str(draft.get("description") or "").strip() or "Credit card",
                "category": draft.get("category") or "Can't classify",
                "project_code": str(draft.get("project_code") or "").strip(),
                "project_name": project_name,
                "amount": float(draft.get("amount") or 0),
                "image_bytes": image_bytes,
                "image_mime": image_mime or "image/jpeg",
            },
        )
        delete_draft(db_path, chat_id)
        return f"Saved credit card — {project_name} — {float(draft.get('amount') or 0):.2f}"

    if not image_bytes:
        return "No photo on this draft. Send the receipt photo again."
    amount = float(draft.get("amount") or 0)
    desc = str(draft.get("description") or "").strip() or "Receipt"
    ref = next_receipt_ref(db_path)
    add_receipt(
        db_path,
        {
            "ref": ref,
            "date": date,
            "description": desc,
            "category": draft.get("category") or "Can't classify",
            "project_code": str(draft.get("project_code") or "").strip(),
            "project_name": project_name,
            "amount": amount,
            "image_bytes": image_bytes,
            "image_mime": image_mime or "image/jpeg",
        },
    )
    delete_draft(db_path, chat_id)
    return f"Saved receipt #{ref} — {desc} — {amount:.2f}"


async def handle_update(db_path, update: dict) -> None:
    callback = update.get("callback_query")
    message = update.get("message") or (callback or {}).get("message")
    from_user = (callback or message or {}).get("from") or {}
    chat = (message or {}).get("chat") or {}
    chat_id = chat.get("id") or (from_user.get("id"))
    user_id = from_user.get("id")
    log.info(
        "telegram update id=%s chat=%s text=%s photo=%s",
        update.get("update_id"),
        chat_id,
        bool((message or {}).get("text")),
        bool(image_file_id(message or {})),
    )

    if chat_id is None:
        return
    if not user_allowed(user_id):
        send_message(chat_id, "This bot is private.")
        return

    if callback:
        data = callback.get("data")
        _api("answerCallbackQuery", callback_query_id=callback["id"])
        await _handle_text(db_path, str(chat_id), "save" if data == "save" else data or "")
        return

    if not message:
        return

    text = (message.get("text") or message.get("caption") or "").strip()
    file_id = image_file_id(message)
    note = parse_telegram_note(text)

    if note.get("kind") == "transport" and not file_id:
        draft = build_transport_draft(note)
        upsert_draft(db_path, str(chat_id), dumps_draft(draft))
        send_message(chat_id, format_draft_message(draft), reply_markup=save_keyboard())
        return

    if note.get("kind") == "credit_card" and not file_id:
        send_message(chat_id, "For credit card, send a photo with caption:\nCC, adnoc")
        return

    if file_id:
        if note.get("kind") == "transport":
            send_message(
                chat_id,
                "Transport does not need a photo. Send text only:\n"
                "TR, Dubai, Abu Dhabi, adnoc\n"
                "Add , return at the end if it is a round trip.",
            )
            return
        send_message(chat_id, "Got the photo. Reading the receipt…")
        image_bytes = download_file(file_id)
        try:
            extracted = await asyncio.to_thread(analyze_receipt_bytes, image_bytes)
        except Exception as exc:
            log.exception("Gemini failed")
            send_message(chat_id, f"Could not read that receipt: {exc}")
            return
        draft = build_receipt_draft(extracted, caption=text)
        upsert_draft(
            db_path,
            str(chat_id),
            dumps_draft(draft),
            image_bytes=image_bytes,
            image_mime="image/jpeg",
        )
        send_message(chat_id, format_draft_message(draft), reply_markup=save_keyboard())
        return

    if text:
        await _handle_text(db_path, str(chat_id), text)
        return

    send_message(chat_id, "Send a receipt photo, or reply with corrections to the current draft.")


async def _handle_text(db_path, chat_id: str, text: str) -> None:
    if is_start_command(text):
        send_message(
            chat_id,
            "How to add expenses:\n\n"
            "Receipts — send a photo. Caption = project name only:\n"
            "adnoc\n\n"
            "Credit card — send a photo, caption:\n"
            "CC, adnoc\n\n"
            "Transport — text only, no photo:\n"
            "TR, Dubai, Abu Dhabi, adnoc\n"
            "TR, Dubai, Abu Dhabi, adnoc, return\n\n"
            "Then tap Save or send save.",
        )
        return

    row = get_draft(db_path, chat_id)
    if not row:
        send_message(chat_id, "No draft yet. Send a receipt photo first.")
        return

    draft = loads_draft(row["payload"])
    image_bytes = row.get("image_bytes")
    image_mime = row.get("image_mime") or "image/jpeg"

    if text.lower() in {"cancel", "/cancel"}:
        delete_draft(db_path, chat_id)
        send_message(chat_id, "Draft cancelled.")
        return

    if parse_save_command(text):
        msg = _save_if_ready(db_path, chat_id, _finalize_description(draft), image_bytes, image_mime)
        send_message(chat_id, msg)
        return

    updated = apply_corrections(draft, text)
    updated = _finalize_description(updated)
    upsert_draft(db_path, chat_id, dumps_draft(updated))
    send_message(chat_id, "Updated.\n\n" + format_draft_message(updated), reply_markup=save_keyboard())


def register_webhook() -> dict | None:
    if not settings.telegram_bot_token or not settings.telegram_webhook_url:
        return None
    url = settings.telegram_webhook_url.rstrip("/") + "/telegram/webhook"
    return _api(
        "setWebhook",
        url=url,
        secret_token=settings.telegram_webhook_secret,
        allowed_updates=["message", "callback_query"],
    )

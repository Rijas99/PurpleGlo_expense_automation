from __future__ import annotations

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.backup import github_backup_enabled, restore_sqlite_backup
from app.bot import handle_update, register_webhook
from app.config import ROOT, settings
from app.constants import CATEGORIES, FOOD_CAP_AMOUNT
from app.corrections import format_display_date, get_meal_description, to_input_date
from app.db import (
    add_colleague,
    add_credit_card,
    add_receipt,
    add_transport,
    archive_current_month,
    archived_month_stats,
    authenticate_user,
    clear_all_drafts,
    clear_current_month,
    delete_archived_month,
    delete_colleague,
    delete_credit_card,
    delete_receipt,
    delete_transport,
    export_sqlite_file,
    get_credit_card,
    get_receipt,
    get_transport,
    get_user,
    import_sqlite_file,
    init_db,
    list_archived_months,
    list_credit_card,
    list_receipts,
    list_transport,
    list_users,
    next_receipt_ref,
    project_codes_for_current,
    project_names,
    turso_enabled,
    update_credit_card,
    update_receipt,
    update_transport,
    working_month_stats,
)
from app.export import (
    build_all_package,
    build_credit_card_package,
    build_receipts_package,
    build_transport_xlsx,
)
from app.gemini import analyze_receipt_bytes
from app.telegram import slugify

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("purpleglo")
logging.getLogger("httpx").setLevel(logging.WARNING)

templates = Jinja2Templates(directory=str(ROOT / "templates"))
STATIC = ROOT / "static"
EXPORTS = ROOT / "data" / "exports"
PENDING = ROOT / "data" / "pending"


def db() -> Path:
    return settings.database_path


MONTH_CHOICES = [
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
MONTH_ABBRS = [abbr for abbr, _ in MONTH_CHOICES]


def current_month_label() -> str:
    now = datetime.now()
    return f"{MONTH_ABBRS[now.month - 1]} {now.year}"


def parse_report_month(label: str | None) -> tuple[str, int]:
    now = datetime.now()
    fallback = (MONTH_ABBRS[now.month - 1], now.year)
    if not label:
        return fallback
    parts = label.strip().split()
    if len(parts) != 2:
        return fallback
    month_part, year_part = parts
    try:
        year = int(year_part)
    except ValueError:
        return fallback
    for abbr, full in MONTH_CHOICES:
        if month_part.lower() in {abbr.lower(), full.lower()}:
            return abbr, year
    return fallback


def format_report_month(month_abbr: str, year: int) -> str:
    abbr = month_abbr if month_abbr in MONTH_ABBRS else datetime.now().strftime("%b")
    return f"{abbr} {year}"


def next_report_month(label: str) -> str:
    abbr, year = parse_report_month(label)
    idx = MONTH_ABBRS.index(abbr)
    if idx == 11:
        return format_report_month("Jan", year + 1)
    return format_report_month(MONTH_ABBRS[idx + 1], year)


def year_choices(selected_year: int) -> list[int]:
    now_year = datetime.now().year
    years = set(range(now_year - 2, now_year + 2))
    years.add(selected_year)
    return sorted(years)


def working_report_month(request: Request) -> str:
    raw = request.session.get("report_month") or current_month_label()
    abbr, year = parse_report_month(raw)
    return format_report_month(abbr, year)


def export_month_label(request: Request) -> str:
    slug = selected_month(request)
    if slug:
        return slug.replace("_", " ")
    return working_report_month(request)


def selected_month(request: Request) -> str | None:
    value = request.query_params.get("month") or request.session.get("month_view")
    if not value or value == "CURRENT":
        return None
    return value


def flash(request: Request, message: str, kind: str = "ok") -> None:
    request.session["flash"] = {"text": message, "kind": kind}


def pop_flash(request: Request) -> dict | None:
    return request.session.pop("flash", None)


def logged_in_user(request: Request) -> dict | None:
    uid = request.session.get("user_id")
    if uid:
        return get_user(db(), int(uid))
    if not settings.app_password:
        people = list_users(db())
        return next((u for u in people if u.get("role") == "admin"), people[0] if people else None)
    return None


def active_user(request: Request) -> dict | None:
    account = logged_in_user(request)
    if not account:
        return None
    if account.get("role") == "admin":
        view_id = request.session.get("view_user_id")
        if view_id:
            other = get_user(db(), int(view_id))
            if other:
                return other
    return account


def owner_id(request: Request) -> int | None:
    user = active_user(request)
    return int(user["id"]) if user else None


def is_admin(request: Request) -> bool:
    user = logged_in_user(request)
    return bool(user and user.get("role") == "admin")


def logged_in(request: Request) -> bool:
    return logged_in_user(request) is not None


def require_auth(request: Request):
    if logged_in(request):
        return None
    return RedirectResponse("/login", status_code=303)


def require_admin(request: Request, message: str = "Only admin can do that."):
    gate = require_auth(request)
    if gate:
        return gate
    if not is_admin(request):
        flash(request, message, "err")
        return RedirectResponse("/manage", status_code=303)
    return None


def export_employee_name(request: Request) -> str:
    user = active_user(request)
    return str((user or {}).get("name") or settings.employee_name)


def pending_path(request: Request, kind: str = "receipt") -> Path:
    sid = request.session.get("sid")
    if not sid:
        sid = secrets.token_hex(8)
        request.session["sid"] = sid
    name = f"{sid}.bin" if kind == "receipt" else f"{sid}-{kind}.bin"
    return PENDING / name


def _capped_desc_amount(description: str, category: str, amount: float, cap_food: str) -> tuple[str, float]:
    final_desc = description.strip()
    final_amt = float(amount)
    if category == "Food & Beverages" and cap_food and final_amt > FOOD_CAP_AMOUNT:
        final_desc = f"{final_desc} (capped at 40)"
        final_amt = FOOD_CAP_AMOUNT
    return final_desc, final_amt


def _date_year(request: Request) -> int:
    return parse_report_month(export_month_label(request))[1]


def _receipt_form_from_row(row: dict, year: int) -> dict:
    return {
        "date": to_input_date(str(row.get("date") or ""), year),
        "description": row.get("description") or "",
        "category": row.get("category") or "",
        "amount": row.get("amount") or 0,
        "project_code": row.get("project_code") or "",
        "project_name": row.get("project_name") or "",
    }


def ctx(request: Request, **extra):
    month_slug = selected_month(request)
    path = request.url.path
    active = "receipts"
    if path.startswith("/credit-card"):
        active = "cc"
    elif path.startswith("/transport"):
        active = "transport"
    elif path.startswith("/manage"):
        active = "manage"
    report_month = working_report_month(request)
    month_abbr, year = parse_report_month(report_month)
    account = logged_in_user(request)
    viewing = active_user(request)
    oid = int(viewing["id"]) if viewing else None
    return {
        "request": request,
        "flash": pop_flash(request),
        "active": active,
        "today": datetime.now().strftime("%Y-%m-%d"),
        "categories": CATEGORIES,
        "project_codes": project_codes_for_current(db(), oid),
        "project_names": project_names(db(), oid),
        "archived": list_archived_months(db(), oid),
        "month_view": month_slug or "CURRENT",
        "report_month": report_month,
        "report_month_month": month_abbr,
        "report_month_year": year,
        "looking_at_label": export_month_label(request),
        "months": MONTH_CHOICES,
        "years": year_choices(year),
        "employee_name": (viewing or {}).get("name") or settings.employee_name,
        "account": account,
        "viewing": viewing,
        "is_admin": bool(account and account.get("role") == "admin"),
        "viewing_other": bool(
            account and viewing and account["id"] != viewing["id"]
        ),
        "gemini_ready": bool(settings.google_api_key),
        "telegram_ready": bool(settings.telegram_bot_token),
        **extra,
    }


async def _poll_telegram():
    from app.bot import TelegramError, _api, delete_webhook

    await asyncio.sleep(0)
    try:
        await asyncio.to_thread(delete_webhook)
        log.info("Telegram webhook cleared; polling")
    except Exception:
        log.exception("Could not clear Telegram webhook")

    offset = 0
    log.info("Telegram polling started (no WEBHOOK_URL)")
    while True:
        try:
            updates = await asyncio.to_thread(
                _api,
                "getUpdates",
                offset=offset,
                timeout=25,
                allowed_updates=["message", "callback_query"],
            )
            for update in updates:
                try:
                    await handle_update(db(), update)
                except Exception:
                    log.exception("Failed handling Telegram update %s", update.get("update_id"))
                offset = int(update["update_id"]) + 1
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            if "409" in str(exc):
                log.warning("Telegram getUpdates conflict; retrying in 3s")
                await asyncio.sleep(3)
                continue
            log.exception("Telegram poll error")
            await asyncio.sleep(3)
        except Exception:
            log.exception("Telegram poll error")
            await asyncio.sleep(3)


def public_base_url() -> str:
    return (
        os.environ.get("KEEP_AWAKE_URL")
        or os.environ.get("RENDER_EXTERNAL_URL")
        or settings.telegram_webhook_url
        or ""
    ).strip().rstrip("/")


async def _keep_awake():
    """Hit the public URL so Render's free instance does not spin down."""
    base = public_base_url()
    if not base:
        return
    await asyncio.sleep(30)
    while True:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.get(f"{base}/health")
            log.info("keep-awake ping ok")
        except Exception:
            log.warning("keep-awake ping failed", exc_info=True)
        await asyncio.sleep(10 * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    restore_sqlite_backup(db())
    init_db(db())
    STATIC.mkdir(exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    PENDING.mkdir(parents=True, exist_ok=True)
    background = []
    if settings.telegram_bot_token and settings.telegram_webhook_url:
        async def _register_webhook_safe():
            try:
                await asyncio.to_thread(register_webhook)
                log.info("Telegram webhook registered")
            except Exception:
                log.exception("Could not register Telegram webhook")

        background.append(asyncio.create_task(_register_webhook_safe()))
    elif settings.telegram_bot_token:
        background.append(asyncio.create_task(_poll_telegram()))
    if public_base_url() and not os.environ.get("PYTEST_CURRENT_TEST"):
        background.append(asyncio.create_task(_keep_awake()))
    yield
    for task in background:
        task.cancel()


app = FastAPI(title="PurpleGlo Expense Manager", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, same_site="lax")
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/health")
def health():
    return {"ok": True, "app": "purpleglo-expense"}


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if logged_in(request) and settings.app_password:
        return RedirectResponse("/", status_code=303)
    if not settings.app_password:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"flash": pop_flash(request)})


@app.post("/login")
def login_submit(request: Request, password: str = Form(...), username: str = Form("")):
    user = None
    name = username.strip()
    if name:
        user = authenticate_user(db(), name, password)
    if user is None:
        user = authenticate_user(db(), "rijas", password)
    if user is None and settings.app_password and password == settings.app_password:
        people = list_users(db())
        user = next((u for u in people if u.get("role") == "admin"), None)
    if user:
        request.session["user_id"] = user["id"]
        request.session["auth"] = True
        request.session.pop("view_user_id", None)
        return RedirectResponse("/", status_code=303)
    flash(request, "Wrong username or password", "err")
    return RedirectResponse("/login", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login" if settings.app_password else "/", status_code=303)


@app.post("/settings")
async def save_settings(
    request: Request,
    report_month_month: str = Form(...),
    report_month_year: int = Form(...),
    month_view: str = Form("CURRENT"),
    next: str = Form(""),
):
    gate = require_auth(request)
    if gate:
        return gate
    request.session["report_month"] = format_report_month(report_month_month, report_month_year)
    request.session["month_view"] = month_view
    dest = next.strip() if next.strip().startswith("/") else ""
    return RedirectResponse(dest or request.headers.get("referer") or "/", status_code=303)


@app.post("/archive")
async def archive_month(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    label = working_report_month(request)
    try:
        archive_current_month(db(), slugify(label), owner_id(request))
        nxt = next_report_month(label)
        request.session["report_month"] = nxt
        request.session["month_view"] = "CURRENT"
        flash(request, f"Archived {label}. Working month is now {nxt}.")
    except Exception as exc:
        flash(request, str(exc), "err")
    return RedirectResponse(request.headers.get("referer") or "/", status_code=303)


def _storage_status() -> dict[str, str]:
    if turso_enabled():
        store = "Turso cloud SQLite"
    else:
        store = "Local SQLite file"
    backup = "GitHub backup on" if github_backup_enabled() else "GitHub backup off"
    return {"store": store, "backup": backup}


@app.get("/manage", response_class=HTMLResponse)
def manage_page(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    stats = working_month_stats(db(), owner_id(request))
    return templates.TemplateResponse(
        request,
        "manage.html",
        ctx(
            request,
            stats=stats,
            archives=archived_month_stats(db(), owner_id(request)),
            storage=_storage_status(),
            people=list_users(db()) if is_admin(request) else [],
        ),
    )


@app.post("/manage/clear-current")
async def manage_clear_current(request: Request, confirm: str = Form("")):
    gate = require_auth(request)
    if gate:
        return gate
    if confirm.strip().upper() != "CLEAR":
        flash(request, "Type CLEAR to confirm.", "err")
        return RedirectResponse("/manage", status_code=303)
    counts = clear_current_month(db(), owner_id(request))
    name = (active_user(request) or {}).get("name") or "this account"
    flash(
        request,
        "Working month cleared "
        f"for {name} "
        f"({counts['receipts']} receipts, {counts['credit_card']} credit cards, "
        f"{counts['transport']} trips). Other people were not touched.",
    )
    return RedirectResponse("/manage", status_code=303)


@app.post("/manage/delete-archive")
async def manage_delete_archive(
    request: Request,
    month_slug: str = Form(...),
    confirm: str = Form(""),
):
    gate = require_auth(request)
    if gate:
        return gate
    if confirm.strip().upper() != "DELETE":
        flash(request, "Type DELETE to confirm.", "err")
        return RedirectResponse("/manage", status_code=303)
    try:
        delete_archived_month(db(), month_slug, owner_id(request))
        flash(request, f"Deleted archive {month_slug.replace('_', ' ')}.")
        if request.session.get("month_view") == month_slug:
            request.session["month_view"] = "CURRENT"
    except Exception as exc:
        flash(request, str(exc), "err")
    return RedirectResponse("/manage", status_code=303)


@app.post("/manage/clear-drafts")
async def manage_clear_drafts(request: Request):
    gate = require_admin(request, "Only admin can clear all Telegram drafts.")
    if gate:
        return gate
    n = clear_all_drafts(db())
    label = "Telegram draft" if n == 1 else "Telegram drafts"
    flash(request, f"Cleared {n} {label}.")
    return RedirectResponse("/manage", status_code=303)


@app.post("/manage/restore-backup")
async def manage_restore_backup(
    request: Request,
    confirm: str = Form(""),
    backup: UploadFile = File(...),
):
    gate = require_admin(request, "Only admin can restore a backup.")
    if gate:
        return gate
    if confirm.strip().upper() != "RESTORE":
        flash(request, "Type RESTORE to confirm.", "err")
        return RedirectResponse("/manage", status_code=303)
    raw = await backup.read()
    if not raw.startswith(b"SQLite format 3"):
        flash(request, "That file is not a SQLite database.", "err")
        return RedirectResponse("/manage", status_code=303)
    dest = PENDING / "restore-upload.db"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    try:
        import_sqlite_file(dest, db())
        stats = working_month_stats(db(), owner_id(request))
        flash(
            request,
            f"Restored backup with {stats['receipts']} working-month receipts.",
        )
    except Exception as exc:
        flash(request, str(exc), "err")
    return RedirectResponse("/manage", status_code=303)


@app.post("/manage/colleagues")
async def manage_add_colleague(
    request: Request,
    name: str = Form(...),
    password: str = Form(...),
):
    gate = require_auth(request)
    if gate:
        return gate
    if not is_admin(request):
        flash(request, "Only admin can add colleagues.", "err")
        return RedirectResponse("/manage", status_code=303)
    try:
        person = add_colleague(db(), name, password)
        flash(
            request,
            f"Added {person['name']} (username {person['username']}). "
            f"Telegram: join {person['invite_code']}",
        )
    except Exception as exc:
        flash(request, str(exc), "err")
    return RedirectResponse("/manage", status_code=303)


@app.post("/manage/view-user")
async def manage_view_user(request: Request, user_id: int = Form(...)):
    gate = require_admin(request, "Only admin can view another person's expenses.")
    if gate:
        return gate
    account = logged_in_user(request)
    if account and int(user_id) == int(account["id"]):
        request.session.pop("view_user_id", None)
        flash(request, "Showing your own expenses.")
        return RedirectResponse("/", status_code=303)
    other = get_user(db(), int(user_id))
    if not other:
        flash(request, "Person not found.", "err")
        return RedirectResponse("/manage", status_code=303)
    request.session["view_user_id"] = other["id"]
    flash(request, f"Viewing {other['name']}'s expenses.")
    return RedirectResponse("/", status_code=303)


@app.post("/manage/delete-colleague")
async def manage_delete_colleague(request: Request, user_id: int = Form(...)):
    gate = require_admin(request, "Only admin can remove colleagues.")
    if gate:
        return gate
    try:
        delete_colleague(db(), int(user_id))
        view_id = request.session.get("view_user_id")
        if view_id and int(view_id) == int(user_id):
            request.session.pop("view_user_id", None)
        flash(request, "Colleague removed.")
    except Exception as exc:
        flash(request, str(exc), "err")
    return RedirectResponse("/manage", status_code=303)


@app.get("/", response_class=HTMLResponse)
def receipts_page(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    month_slug = selected_month(request)
    rows = list_receipts(db(), month_slug, owner_id(request))
    total = sum(float(r["amount"] or 0) for r in rows)
    analysis = request.session.get("receipt_analysis")
    return templates.TemplateResponse(
        request,
        "receipts.html",
        ctx(
            request,
            rows=rows,
            total=total,
            analysis=analysis,
            food_cap=FOOD_CAP_AMOUNT,
            editing=None,
        ),
    )


@app.post("/receipts/analyze")
async def analyze_receipt(request: Request, photo: UploadFile = File(...)):
    gate = require_auth(request)
    if gate:
        return gate
    raw = await photo.read()
    if not raw:
        flash(request, "Empty file", "err")
        return RedirectResponse("/", status_code=303)
    try:
        data = analyze_receipt_bytes(raw)
        if data.get("category") == "Food & Beverages" and not data.get("description"):
            data["description"] = get_meal_description(data.get("time"))
        pending_path(request).write_bytes(raw)
        request.session["receipt_analysis"] = data
        request.session["receipt_mime"] = photo.content_type or "image/jpeg"
        flash(request, "Receipt read. Check the fields and save.")
    except Exception as exc:
        pending_path(request).write_bytes(raw)
        request.session["receipt_analysis"] = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "amount": 0,
            "description": "",
            "category": "Can't classify",
        }
        request.session["receipt_mime"] = photo.content_type or "image/jpeg"
        flash(request, f"Could not auto-read receipt ({exc}). Fill the form manually.", "err")
    return RedirectResponse("/", status_code=303)


@app.post("/receipts/save")
async def save_receipt(
    request: Request,
    date: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    project_code: str = Form(""),
    project_name: str = Form(...),
    amount: float = Form(...),
    cap_food: str = Form(""),
):
    gate = require_auth(request)
    if gate:
        return gate
    if selected_month(request):
        flash(request, "Switch to CURRENT month to add receipts.", "err")
        return RedirectResponse("/", status_code=303)
    if not project_name.strip():
        flash(request, "Project name is required.", "err")
        return RedirectResponse("/", status_code=303)

    final_desc, final_amt = _capped_desc_amount(description, category, amount, cap_food)

    img_file = pending_path(request)
    image_bytes = img_file.read_bytes() if img_file.exists() else None
    if img_file.exists():
        img_file.unlink()
    image_mime = request.session.pop("receipt_mime", "image/jpeg")
    request.session.pop("receipt_analysis", None)

    add_receipt(
        db(),
        {
            "ref": next_receipt_ref(db(), owner_id(request)),
            "date": format_display_date(date),
            "description": final_desc,
            "category": category,
            "project_code": project_code.strip(),
            "project_name": project_name.strip(),
            "amount": final_amt,
            "image_bytes": image_bytes,
            "image_mime": image_mime,
            "owner_id": owner_id(request),
        },
    )
    flash(request, "Receipt saved.")
    return RedirectResponse("/", status_code=303)


@app.get("/receipts/{ref}/edit")
def receipts_edit(request: Request, ref: int):
    gate = require_auth(request)
    if gate:
        return gate
    if selected_month(request):
        flash(request, "Archived months are read-only.", "err")
        return RedirectResponse("/", status_code=303)
    row = get_receipt(db(), ref, None, owner_id(request))
    if not row:
        flash(request, "Receipt not found.", "err")
        return RedirectResponse("/", status_code=303)
    rows = list_receipts(db(), None, owner_id(request))
    total = sum(float(r["amount"] or 0) for r in rows)
    return templates.TemplateResponse(
        request,
        "receipts.html",
        ctx(
            request,
            rows=rows,
            total=total,
            analysis=_receipt_form_from_row(row, _date_year(request)),
            food_cap=FOOD_CAP_AMOUNT,
            editing=row,
        ),
    )


@app.post("/receipts/{ref}/update")
async def receipts_update(
    request: Request,
    ref: int,
    date: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    project_code: str = Form(""),
    project_name: str = Form(...),
    amount: float = Form(...),
    cap_food: str = Form(""),
    photo: UploadFile | None = File(None),
):
    gate = require_auth(request)
    if gate:
        return gate
    if selected_month(request):
        flash(request, "Archived months are read-only.", "err")
        return RedirectResponse("/", status_code=303)
    if not project_name.strip():
        flash(request, "Project name is required.", "err")
        return RedirectResponse("/", status_code=303)
    final_desc, final_amt = _capped_desc_amount(description, category, amount, cap_food)
    payload = {
        "date": format_display_date(date),
        "description": final_desc,
        "category": category,
        "project_code": project_code.strip(),
        "project_name": project_name.strip(),
        "amount": final_amt,
    }
    replace_image = bool(photo and photo.filename)
    if replace_image:
        payload["image_bytes"] = await photo.read()
        payload["image_mime"] = photo.content_type or "image/jpeg"
    if update_receipt(db(), ref, payload, replace_image=replace_image, owner_id=owner_id(request)):
        flash(request, f"Updated receipt {ref}.")
    else:
        flash(request, "Receipt not found.", "err")
    return RedirectResponse("/", status_code=303)


@app.post("/receipts/{ref}/delete")
async def receipts_delete(request: Request, ref: int):
    gate = require_auth(request)
    if gate:
        return gate
    if selected_month(request):
        flash(request, "Archived months are read-only.", "err")
        return RedirectResponse("/", status_code=303)
    if delete_receipt(db(), ref, owner_id(request)):
        flash(request, f"Deleted receipt {ref}.")
    else:
        flash(request, "Receipt not found.", "err")
    return RedirectResponse("/", status_code=303)


@app.get("/credit-card", response_class=HTMLResponse)
def credit_page(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    rows = list_credit_card(db(), selected_month(request), owner_id(request))
    total = sum(float(r["amount"] or 0) for r in rows)
    return templates.TemplateResponse(
        request,
        "credit_card.html",
        ctx(
            request,
            rows=rows,
            total=total,
            analysis=request.session.get("cc_analysis"),
            food_cap=FOOD_CAP_AMOUNT,
            editing=None,
        ),
    )


@app.post("/credit-card/analyze")
async def credit_analyze(request: Request, photo: UploadFile = File(...)):
    gate = require_auth(request)
    if gate:
        return gate
    raw = await photo.read()
    if not raw:
        flash(request, "Empty file", "err")
        return RedirectResponse("/credit-card", status_code=303)
    try:
        data = analyze_receipt_bytes(raw)
        if data.get("category") == "Food & Beverages" and not data.get("description"):
            data["description"] = get_meal_description(data.get("time"))
        pending_path(request, "cc").write_bytes(raw)
        request.session["cc_analysis"] = data
        request.session["cc_mime"] = photo.content_type or "image/jpeg"
        flash(request, "Receipt read. Check the fields and save.")
    except Exception as exc:
        pending_path(request, "cc").write_bytes(raw)
        request.session["cc_analysis"] = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "amount": 0,
            "description": "",
            "category": "Can't classify",
        }
        request.session["cc_mime"] = photo.content_type or "image/jpeg"
        flash(request, f"Could not auto-read receipt ({exc}). Fill the form manually.", "err")
    return RedirectResponse("/credit-card", status_code=303)


@app.post("/credit-card/save")
async def credit_save(
    request: Request,
    date: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    project_code: str = Form(""),
    project_name: str = Form(...),
    amount: float = Form(...),
    cap_food: str = Form(""),
    photo: UploadFile | None = File(None),
):
    gate = require_auth(request)
    if gate:
        return gate
    if selected_month(request):
        flash(request, "Switch to CURRENT month to add expenses.", "err")
        return RedirectResponse("/credit-card", status_code=303)
    final_desc, final_amt = _capped_desc_amount(description, category, amount, cap_food)
    image_bytes = None
    image_mime = "image/jpeg"
    if photo and photo.filename:
        image_bytes = await photo.read()
        image_mime = photo.content_type or "image/jpeg"
    else:
        img_file = pending_path(request, "cc")
        if img_file.exists():
            image_bytes = img_file.read_bytes()
            image_mime = request.session.get("cc_mime") or "image/jpeg"
            img_file.unlink()
    request.session.pop("cc_analysis", None)
    request.session.pop("cc_mime", None)
    add_credit_card(
        db(),
        {
            "date": format_display_date(date),
            "description": final_desc,
            "category": category,
            "project_code": project_code.strip(),
            "project_name": project_name.strip(),
            "amount": final_amt,
            "image_bytes": image_bytes,
            "image_mime": image_mime,
            "owner_id": owner_id(request),
        },
    )
    flash(request, "Credit card expense saved.")
    return RedirectResponse("/credit-card", status_code=303)


@app.get("/credit-card/{row_id}/edit")
def credit_edit(request: Request, row_id: int):
    gate = require_auth(request)
    if gate:
        return gate
    if selected_month(request):
        flash(request, "Archived months are read-only.", "err")
        return RedirectResponse("/credit-card", status_code=303)
    row = get_credit_card(db(), row_id, None, owner_id(request))
    if not row:
        flash(request, "Expense not found.", "err")
        return RedirectResponse("/credit-card", status_code=303)
    rows = list_credit_card(db(), None, owner_id(request))
    total = sum(float(r["amount"] or 0) for r in rows)
    return templates.TemplateResponse(
        request,
        "credit_card.html",
        ctx(
            request,
            rows=rows,
            total=total,
            analysis=_receipt_form_from_row(row, _date_year(request)),
            food_cap=FOOD_CAP_AMOUNT,
            editing=row,
        ),
    )


@app.post("/credit-card/{row_id}/update")
async def credit_update(
    request: Request,
    row_id: int,
    date: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    project_code: str = Form(""),
    project_name: str = Form(...),
    amount: float = Form(...),
    cap_food: str = Form(""),
    photo: UploadFile | None = File(None),
):
    gate = require_auth(request)
    if gate:
        return gate
    if selected_month(request):
        flash(request, "Archived months are read-only.", "err")
        return RedirectResponse("/credit-card", status_code=303)
    final_desc, final_amt = _capped_desc_amount(description, category, amount, cap_food)
    payload = {
        "date": format_display_date(date),
        "description": final_desc,
        "category": category,
        "project_code": project_code.strip(),
        "project_name": project_name.strip(),
        "amount": final_amt,
    }
    replace_image = bool(photo and photo.filename)
    if replace_image:
        payload["image_bytes"] = await photo.read()
        payload["image_mime"] = photo.content_type or "image/jpeg"
    if update_credit_card(db(), row_id, payload, replace_image=replace_image, owner_id=owner_id(request)):
        flash(request, "Updated credit card expense.")
    else:
        flash(request, "Expense not found.", "err")
    return RedirectResponse("/credit-card", status_code=303)


@app.post("/credit-card/{row_id}/delete")
async def credit_delete(request: Request, row_id: int):
    gate = require_auth(request)
    if gate:
        return gate
    delete_credit_card(db(), row_id, owner_id(request))
    flash(request, "Deleted.")
    return RedirectResponse("/credit-card", status_code=303)


@app.get("/transport", response_class=HTMLResponse)
def transport_page(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    rows = list_transport(db(), selected_month(request), owner_id(request))
    return templates.TemplateResponse(
        request, "transport.html", ctx(request, rows=rows, editing=None)
    )


@app.post("/transport/save")
async def transport_save(
    request: Request,
    date: str = Form(...),
    from_location: str = Form(...),
    destination: str = Form(...),
    project_code: str = Form(""),
    project_name: str = Form(...),
    return_included: str = Form(""),
):
    gate = require_auth(request)
    if gate:
        return gate
    if selected_month(request):
        flash(request, "Switch to CURRENT month to add expenses.", "err")
        return RedirectResponse("/transport", status_code=303)
    add_transport(
        db(),
        {
            "date": format_display_date(date),
            "from_location": from_location.strip(),
            "destination": destination.strip(),
            "return_included": bool(return_included),
            "project_code": project_code.strip(),
            "project_name": project_name.strip(),
            "owner_id": owner_id(request),
        },
    )
    flash(request, "Transport expense saved.")
    return RedirectResponse("/transport", status_code=303)


@app.get("/transport/{row_id}/edit")
def transport_edit(request: Request, row_id: int):
    gate = require_auth(request)
    if gate:
        return gate
    if selected_month(request):
        flash(request, "Archived months are read-only.", "err")
        return RedirectResponse("/transport", status_code=303)
    row = get_transport(db(), row_id, None, owner_id(request))
    if not row:
        flash(request, "Trip not found.", "err")
        return RedirectResponse("/transport", status_code=303)
    rows = list_transport(db(), None, owner_id(request))
    form = {
        "date": to_input_date(str(row.get("date") or ""), _date_year(request)),
        "from_location": row.get("from_location") or "",
        "destination": row.get("destination") or "",
        "return_included": bool(row.get("return_included")),
        "project_code": row.get("project_code") or "",
        "project_name": row.get("project_name") or "",
    }
    return templates.TemplateResponse(
        request, "transport.html", ctx(request, rows=rows, editing=row, analysis=form)
    )


@app.post("/transport/{row_id}/update")
async def transport_update(
    request: Request,
    row_id: int,
    date: str = Form(...),
    from_location: str = Form(...),
    destination: str = Form(...),
    project_code: str = Form(""),
    project_name: str = Form(...),
    return_included: str = Form(""),
):
    gate = require_auth(request)
    if gate:
        return gate
    if selected_month(request):
        flash(request, "Archived months are read-only.", "err")
        return RedirectResponse("/transport", status_code=303)
    ok = update_transport(
        db(),
        row_id,
        {
            "date": format_display_date(date),
            "from_location": from_location.strip(),
            "destination": destination.strip(),
            "return_included": bool(return_included),
            "project_code": project_code.strip(),
            "project_name": project_name.strip(),
        },
        owner_id=owner_id(request),
    )
    flash(request, "Updated trip." if ok else "Trip not found.", "ok" if ok else "err")
    return RedirectResponse("/transport", status_code=303)


@app.post("/transport/{row_id}/delete")
async def transport_delete(request: Request, row_id: int):
    gate = require_auth(request)
    if gate:
        return gate
    delete_transport(db(), row_id, owner_id(request))
    flash(request, "Deleted.")
    return RedirectResponse("/transport", status_code=303)


@app.get("/export/receipts")
def export_receipts(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    month = export_month_label(request)
    month_slug = selected_month(request)
    try:
        zip_path = build_receipts_package(
            db(),
            month=month,
            month_slug=month_slug,
            work_dir=EXPORTS,
            template_path=settings.template_path,
            employee_name=export_employee_name(request),
            owner_id=owner_id(request),
        )
    except Exception as exc:
        flash(request, str(exc), "err")
        return RedirectResponse("/", status_code=303)
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")


@app.get("/export/credit-card")
def export_cc(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    month = export_month_label(request)
    month_slug = selected_month(request)
    try:
        zip_path = build_credit_card_package(
            db(), month, month_slug, EXPORTS, owner_id=owner_id(request)
        )
    except Exception as exc:
        flash(request, str(exc), "err")
        return RedirectResponse("/credit-card", status_code=303)
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")


@app.get("/export/all")
def export_all(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    month = export_month_label(request)
    month_slug = selected_month(request)
    try:
        zip_path = build_all_package(
            db(),
            month=month,
            month_slug=month_slug,
            work_dir=EXPORTS,
            template_path=settings.template_path,
            employee_name=export_employee_name(request),
            owner_id=owner_id(request),
        )
    except Exception as exc:
        flash(request, str(exc), "err")
        return RedirectResponse(request.headers.get("referer") or "/", status_code=303)
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")


@app.get("/export/transport")
def export_transport(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    slug = selected_month(request) or slugify(export_month_label(request))
    dest = EXPORTS / f"Transport_{slug}.xlsx"
    try:
        build_transport_xlsx(db(), selected_month(request), dest, owner_id=owner_id(request))
    except Exception as exc:
        flash(request, str(exc), "err")
        return RedirectResponse("/transport", status_code=303)
    return FileResponse(
        dest,
        filename=dest.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/backup")
def backup(request: Request):
    gate = require_admin(request, "Only admin can download the full SQLite backup.")
    if gate:
        return gate
    dest = EXPORTS / "expenses.db"
    try:
        export_sqlite_file(dest)
    except Exception as exc:
        flash(request, str(exc), "err")
        return RedirectResponse(request.headers.get("referer") or "/", status_code=303)
    return FileResponse(dest, filename="expenses.db", media_type="application/octet-stream")


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if settings.telegram_webhook_secret and secret != settings.telegram_webhook_secret:
        return Response(status_code=403)
    update = await request.json()
    background_tasks.add_task(handle_update, db(), update)
    return {"ok": True}


@app.get("/receipts/{ref}/image")
def receipt_image(request: Request, ref: int):
    gate = require_auth(request)
    if gate:
        return gate
    row = get_receipt(db(), ref, selected_month(request), owner_id(request))
    if row and row.get("image_bytes"):
        return Response(content=row["image_bytes"], media_type=row.get("image_mime") or "image/jpeg")
    return Response(status_code=404)


@app.get("/credit-card/{row_id}/image")
def credit_image(request: Request, row_id: int):
    gate = require_auth(request)
    if gate:
        return gate
    row = get_credit_card(db(), row_id, selected_month(request), owner_id(request))
    if row and row.get("image_bytes"):
        return Response(content=row["image_bytes"], media_type=row.get("image_mime") or "image/jpeg")
    return Response(status_code=404)

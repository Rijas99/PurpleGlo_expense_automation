from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.bot import handle_update, register_webhook
from app.config import ROOT, settings
from app.constants import CATEGORIES, FOOD_CAP_AMOUNT
from app.corrections import format_display_date, get_meal_description, to_input_date
from app.db import (
    add_credit_card,
    add_receipt,
    add_transport,
    archive_current_month,
    delete_credit_card,
    delete_receipt,
    delete_transport,
    get_credit_card,
    get_receipt,
    get_transport,
    init_db,
    list_archived_months,
    list_credit_card,
    list_receipts,
    list_transport,
    next_receipt_ref,
    project_codes_for_current,
    project_names,
    update_credit_card,
    update_receipt,
    update_transport,
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


def logged_in(request: Request) -> bool:
    if not settings.app_password:
        return True
    return bool(request.session.get("auth"))


def require_auth(request: Request):
    if logged_in(request):
        return None
    return RedirectResponse("/login", status_code=303)


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
    report_month = working_report_month(request)
    month_abbr, year = parse_report_month(report_month)
    return {
        "request": request,
        "flash": pop_flash(request),
        "active": active,
        "today": datetime.now().strftime("%Y-%m-%d"),
        "categories": CATEGORIES,
        "project_codes": project_codes_for_current(db()),
        "project_names": project_names(db()),
        "archived": list_archived_months(db()),
        "month_view": month_slug or "CURRENT",
        "report_month": report_month,
        "report_month_month": month_abbr,
        "report_month_year": year,
        "looking_at_label": export_month_label(request),
        "months": MONTH_CHOICES,
        "years": year_choices(year),
        "employee_name": settings.employee_name,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(db())
    STATIC.mkdir(exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    PENDING.mkdir(parents=True, exist_ok=True)
    poll_task = None
    if settings.telegram_bot_token and settings.telegram_webhook_url:
        try:
            register_webhook()
            log.info("Telegram webhook registered")
        except Exception:
            log.exception("Could not register Telegram webhook")
    elif settings.telegram_bot_token:
        poll_task = asyncio.create_task(_poll_telegram())
    yield
    if poll_task:
        poll_task.cancel()


app = FastAPI(title="PurpleGlo Expense Manager", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, same_site="lax")
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/health")
def health():
    return {"ok": True, "app": "purpleglo-expense"}


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if not settings.app_password:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"flash": pop_flash(request)})


@app.post("/login")
def login_submit(request: Request, password: str = Form(...)):
    if password == settings.app_password:
        request.session["auth"] = True
        return RedirectResponse("/", status_code=303)
    flash(request, "Wrong password", "err")
    return RedirectResponse("/login", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.post("/settings")
async def save_settings(
    request: Request,
    report_month_month: str = Form(...),
    report_month_year: int = Form(...),
    month_view: str = Form("CURRENT"),
):
    gate = require_auth(request)
    if gate:
        return gate
    request.session["report_month"] = format_report_month(report_month_month, report_month_year)
    request.session["month_view"] = month_view
    return RedirectResponse(request.headers.get("referer") or "/", status_code=303)


@app.post("/archive")
async def archive_month(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    label = working_report_month(request)
    try:
        archive_current_month(db(), slugify(label))
        nxt = next_report_month(label)
        request.session["report_month"] = nxt
        request.session["month_view"] = "CURRENT"
        flash(request, f"Archived {label}. Working month is now {nxt}.")
    except Exception as exc:
        flash(request, str(exc), "err")
    return RedirectResponse("/", status_code=303)


@app.get("/", response_class=HTMLResponse)
def receipts_page(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    month_slug = selected_month(request)
    rows = list_receipts(db(), month_slug)
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
            "ref": next_receipt_ref(db()),
            "date": format_display_date(date),
            "description": final_desc,
            "category": category,
            "project_code": project_code.strip(),
            "project_name": project_name.strip(),
            "amount": final_amt,
            "image_bytes": image_bytes,
            "image_mime": image_mime,
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
    row = get_receipt(db(), ref, None)
    if not row:
        flash(request, "Receipt not found.", "err")
        return RedirectResponse("/", status_code=303)
    rows = list_receipts(db(), None)
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
    if update_receipt(db(), ref, payload, replace_image=replace_image):
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
    if delete_receipt(db(), ref):
        flash(request, f"Deleted receipt {ref}.")
    else:
        flash(request, "Receipt not found.", "err")
    return RedirectResponse("/", status_code=303)


@app.get("/credit-card", response_class=HTMLResponse)
def credit_page(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    rows = list_credit_card(db(), selected_month(request))
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
    row = get_credit_card(db(), row_id, None)
    if not row:
        flash(request, "Expense not found.", "err")
        return RedirectResponse("/credit-card", status_code=303)
    rows = list_credit_card(db(), None)
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
    if update_credit_card(db(), row_id, payload, replace_image=replace_image):
        flash(request, "Updated credit card expense.")
    else:
        flash(request, "Expense not found.", "err")
    return RedirectResponse("/credit-card", status_code=303)


@app.post("/credit-card/{row_id}/delete")
async def credit_delete(request: Request, row_id: int):
    gate = require_auth(request)
    if gate:
        return gate
    delete_credit_card(db(), row_id)
    flash(request, "Deleted.")
    return RedirectResponse("/credit-card", status_code=303)


@app.get("/transport", response_class=HTMLResponse)
def transport_page(request: Request):
    gate = require_auth(request)
    if gate:
        return gate
    rows = list_transport(db(), selected_month(request))
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
    row = get_transport(db(), row_id, None)
    if not row:
        flash(request, "Trip not found.", "err")
        return RedirectResponse("/transport", status_code=303)
    rows = list_transport(db(), None)
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
    )
    flash(request, "Updated trip." if ok else "Trip not found.", "ok" if ok else "err")
    return RedirectResponse("/transport", status_code=303)


@app.post("/transport/{row_id}/delete")
async def transport_delete(request: Request, row_id: int):
    gate = require_auth(request)
    if gate:
        return gate
    delete_transport(db(), row_id)
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
            employee_name=settings.employee_name,
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
        zip_path = build_credit_card_package(db(), month, month_slug, EXPORTS)
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
            employee_name=settings.employee_name,
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
        build_transport_xlsx(db(), selected_month(request), dest)
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
    gate = require_auth(request)
    if gate:
        return gate
    path = db()
    if not path.exists():
        flash(request, "No database yet.", "err")
        return RedirectResponse("/", status_code=303)
    return FileResponse(path, filename="expenses.db", media_type="application/octet-stream")


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if settings.telegram_webhook_secret and secret != settings.telegram_webhook_secret:
        return Response(status_code=403)
    update = await request.json()
    await handle_update(db(), update)
    return {"ok": True}


@app.get("/receipts/{ref}/image")
def receipt_image(request: Request, ref: int):
    gate = require_auth(request)
    if gate:
        return gate
    row = get_receipt(db(), ref, selected_month(request))
    if row and row.get("image_bytes"):
        return Response(content=row["image_bytes"], media_type=row.get("image_mime") or "image/jpeg")
    return Response(status_code=404)


@app.get("/credit-card/{row_id}/image")
def credit_image(request: Request, row_id: int):
    gate = require_auth(request)
    if gate:
        return gate
    row = get_credit_card(db(), row_id, selected_month(request))
    if row and row.get("image_bytes"):
        return Response(content=row["image_bytes"], media_type=row.get("image_mime") or "image/jpeg")
    return Response(status_code=404)

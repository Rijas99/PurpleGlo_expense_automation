from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from openpyxl import Workbook

from app.db import list_credit_card, list_receipts, list_transport
from app.excel_fill import fill_expense_form, project_code_for_form


def _simple_xlsx(rows: list[dict], headers: list[str], dest: Path) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h, "") for h in headers])
    dest.parent.mkdir(parents=True, exist_ok=True)
    wb.save(dest)
    return dest


def _zip_folder(folder: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in folder.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(folder))
    return zip_path


def _write_bills(rows: list[dict], bills_dir: Path) -> None:
    bills_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(rows, start=1):
        blob = rec.get("image_bytes")
        if not blob:
            continue
        name = rec.get("ref") if rec.get("ref") not in (None, "") else i
        (bills_dir / f"{name}.jpg").write_bytes(blob)


def _month_slug(month: str, month_slug: str | None) -> str:
    return month_slug or month.replace(" ", "_")


def write_receipts_folder(
    db_path: Path | str,
    month: str,
    month_slug: str | None,
    dest_dir: Path,
    template_path: Path,
    employee_name: str,
) -> Path:
    receipts = list_receipts(db_path, month_slug)
    if not receipts:
        raise RuntimeError("No receipts for this month.")
    slug = _month_slug(month, month_slug)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fill_expense_form(
        receipts,
        month=month,
        dest=dest_dir / f"Expense_Form_{slug}.xlsx",
        template_path=template_path,
        employee_name=employee_name,
    )
    _write_bills(receipts, dest_dir / f"bills_{slug}")
    return dest_dir


def build_receipts_package(
    db_path: Path | str,
    month: str,
    month_slug: str | None,
    work_dir: Path,
    template_path: Path,
    employee_name: str,
) -> Path:
    slug = _month_slug(month, month_slug)
    out_dir = work_dir / f"receipts_{slug}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    write_receipts_folder(db_path, month, month_slug, out_dir, template_path, employee_name)
    return _zip_folder(out_dir, work_dir / f"Receipts_Submission_{slug}.zip")


def write_credit_card_folder(
    db_path: Path | str, month: str, month_slug: str | None, dest_dir: Path
) -> Path:
    rows = list_credit_card(db_path, month_slug)
    if not rows:
        raise RuntimeError("No credit card expenses for this month.")
    slug = _month_slug(month, month_slug)
    dest_dir.mkdir(parents=True, exist_ok=True)
    mapped = [
        {
            "Ref": r.get("ref") or "",
            "Date": r["date"],
            "Description": r["description"],
            "Category": r["category"],
            "Project Code": project_code_for_form(r),
            "Project Name": r.get("project_name") or "",
            "Amount": r["amount"],
        }
        for r in rows
    ]
    _simple_xlsx(
        mapped,
        ["Ref", "Date", "Description", "Category", "Project Code", "Project Name", "Amount"],
        dest_dir / f"CreditCard_{slug}.xlsx",
    )
    _write_bills(rows, dest_dir / f"bills_{slug}")
    return dest_dir


def build_credit_card_package(
    db_path: Path | str, month: str, month_slug: str | None, work_dir: Path
) -> Path:
    slug = _month_slug(month, month_slug)
    out_dir = work_dir / f"credit_{slug}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    write_credit_card_folder(db_path, month, month_slug, out_dir)
    return _zip_folder(out_dir, work_dir / f"CreditCard_Submission_{slug}.zip")


def write_transport_folder(
    db_path: Path | str, month: str, month_slug: str | None, dest_dir: Path
) -> Path:
    rows = list_transport(db_path, month_slug)
    if not rows:
        raise RuntimeError("No transport expenses for this month.")
    slug = _month_slug(month, month_slug)
    dest_dir.mkdir(parents=True, exist_ok=True)
    mapped = [
        {
            "Date": r["date"],
            "From": r["from_location"],
            "Destination": r["destination"],
            "Return Included": "Return Included" if r.get("return_included") else "",
            "Project Code": project_code_for_form(r),
            "Project Name": r.get("project_name") or "",
        }
        for r in rows
    ]
    _simple_xlsx(
        mapped,
        ["Date", "From", "Destination", "Return Included", "Project Code", "Project Name"],
        dest_dir / f"Transport_{slug}.xlsx",
    )
    return dest_dir


def build_transport_xlsx(
    db_path: Path | str, month_slug: str | None, dest: Path, month: str = ""
) -> Path:
    label = month or "export"
    tmp = dest.parent / f"_transport_{_month_slug(label, month_slug)}"
    if tmp.exists():
        shutil.rmtree(tmp)
    write_transport_folder(db_path, label, month_slug, tmp)
    xlsx = next(tmp.glob("*.xlsx"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xlsx, dest)
    shutil.rmtree(tmp)
    return dest


def build_all_package(
    db_path: Path | str,
    month: str,
    month_slug: str | None,
    work_dir: Path,
    template_path: Path,
    employee_name: str,
) -> Path:
    slug = _month_slug(month, month_slug)
    out_dir = work_dir / f"all_{slug}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    try:
        write_receipts_folder(
            db_path, month, month_slug, out_dir / "receipts", template_path, employee_name
        )
        added += 1
    except RuntimeError:
        pass
    try:
        write_credit_card_folder(db_path, month, month_slug, out_dir / "credit_card")
        added += 1
    except RuntimeError:
        pass
    try:
        write_transport_folder(db_path, month, month_slug, out_dir / "transport")
        added += 1
    except RuntimeError:
        pass
    if added == 0:
        raise RuntimeError("Nothing to download for this month.")
    return _zip_folder(out_dir, work_dir / f"All_Expenses_{slug}.zip")

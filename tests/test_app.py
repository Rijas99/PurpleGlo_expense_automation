from fastapi.testclient import TestClient

from app import config
from app.db import add_credit_card, add_receipt, add_transport
from app.main import app, next_report_month, public_base_url


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["ok"] is True


def test_public_base_url_uses_webhook_setting(monkeypatch):
    monkeypatch.delenv("KEEP_AWAKE_URL", raising=False)
    monkeypatch.delenv("RENDER_EXTERNAL_URL", raising=False)
    monkeypatch.setattr(config.settings, "telegram_webhook_url", "https://purpleglo-expense.onrender.com/")
    assert public_base_url() == "https://purpleglo-expense.onrender.com"


def test_telegram_webhook_returns_ok_in_background(monkeypatch):
    monkeypatch.setattr(config.settings, "telegram_webhook_secret", "test-secret")
    seen = {}

    async def fake_handle(db_path, update):
        seen["update"] = update

    monkeypatch.setattr("app.main.handle_update", fake_handle)
    with TestClient(app) as client:
        response = client.post(
            "/telegram/webhook",
            json={"update_id": 7, "message": {"text": "hi"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert seen["update"]["update_id"] == 7


def test_home_renders_receipts_page():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Receipts" in response.text
        assert "PurpleGlo" in response.text


def test_sidebar_uses_month_and_year_selects():
    with TestClient(app) as client:
        response = client.get("/")
        assert 'name="report_month_month"' in response.text
        assert 'name="report_month_year"' in response.text
        assert 'name="report_month"' not in response.text.replace("report_month_month", "").replace("report_month_year", "")
        assert ">Apply<" not in response.text
        assert "Look at" not in response.text


def test_save_settings_sets_report_month_from_selects(isolated_db):
    with TestClient(app) as client:
        response = client.post(
            "/settings",
            data={
                "report_month_month": "Sep",
                "report_month_year": "2026",
                "month_view": "CURRENT",
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert 'value="Sep" selected' in response.text
        assert 'value="2026" selected' in response.text


def test_save_receipt_then_list(isolated_db):
    with TestClient(app) as client:
        response = client.post(
            "/receipts/save",
            data={
                "date": "2026-08-15",
                "description": "Lunch",
                "category": "Food & Beverages",
                "project_code": "P1",
                "project_name": "Alpha",
                "amount": "40",
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert "Lunch" in response.text
        assert "Alpha" in response.text


def test_export_zip_uses_template(isolated_db, tmp_path):
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "15-Aug",
            "description": "Lunch",
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "Alpha",
            "amount": 40,
            "image_bytes": b"not-a-real-jpeg",
            "image_mime": "image/jpeg",
        },
    )
    with TestClient(app) as client:
        response = client.get("/export/receipts")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/zip")
        assert response.content[:2] == b"PK"


def test_credit_card_zip_includes_bill_photo(isolated_db):
    add_credit_card(
        isolated_db,
        {
            "date": "15-Aug",
            "description": "AWS",
            "category": "Subscriptions",
            "project_code": "",
            "project_name": "adnoc",
            "amount": 20,
            "image_bytes": b"cc-photo",
            "image_mime": "image/jpeg",
        },
    )
    with TestClient(app) as client:
        response = client.get("/export/credit-card")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/zip")
        import io
        import zipfile

        names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
        assert any(n.endswith("1.jpg") for n in names)
        assert any(n.endswith(".xlsx") for n in names)


def test_download_all_includes_receipts_cc_and_transport(isolated_db):
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "15-Aug",
            "description": "Lunch",
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "Alpha",
            "amount": 40,
            "image_bytes": b"r",
            "image_mime": "image/jpeg",
        },
    )
    add_credit_card(
        isolated_db,
        {
            "date": "16-Aug",
            "description": "AWS",
            "category": "Subscriptions",
            "project_code": "",
            "project_name": "adnoc",
            "amount": 20,
            "image_bytes": b"c",
            "image_mime": "image/jpeg",
        },
    )
    add_transport(
        isolated_db,
        {
            "date": "17-Aug",
            "from_location": "Dubai",
            "destination": "Abu Dhabi",
            "return_included": True,
            "project_code": "",
            "project_name": "adnoc",
        },
    )
    with TestClient(app) as client:
        response = client.get("/export/all")
        assert response.status_code == 200
        import io
        import zipfile

        names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
        joined = " ".join(names).replace("\\", "/")
        assert "receipts/" in joined
        assert "credit_card/" in joined
        assert "transport/" in joined


def test_next_report_month_rolls_year():
    assert next_report_month("Aug 2026") == "Sep 2026"
    assert next_report_month("Dec 2026") == "Jan 2027"


def test_archive_advances_working_month(isolated_db):
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "15-Aug",
            "description": "Lunch",
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "Alpha",
            "amount": 40,
        },
    )
    with TestClient(app) as client:
        client.post(
            "/settings",
            data={
                "report_month_month": "Aug",
                "report_month_year": "2026",
                "month_view": "CURRENT",
            },
        )
        response = client.post("/archive", follow_redirects=True)
        assert response.status_code == 200
        assert 'value="Sep" selected' in response.text
        assert 'value="2026" selected' in response.text
        assert "Look at" in response.text
        assert "Aug 2026" in response.text


def test_download_all_uses_looked_at_archived_month(isolated_db):
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "15-Aug",
            "description": "August lunch",
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "Alpha",
            "amount": 40,
            "image_bytes": b"aug",
            "image_mime": "image/jpeg",
            "month_slug": "Aug_2026",
        },
    )
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "September coffee",
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "Alpha",
            "amount": 12,
            "image_bytes": b"sep",
            "image_mime": "image/jpeg",
        },
    )
    with TestClient(app) as client:
        client.post(
            "/settings",
            data={
                "report_month_month": "Sep",
                "report_month_year": "2026",
                "month_view": "Aug_2026",
            },
        )
        response = client.get("/export/all")
        assert response.status_code == 200
        disposition = response.headers.get("content-disposition", "")
        assert "Aug_2026" in disposition
        import io
        import zipfile

        names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
        joined = " ".join(names).replace("\\", "/")
        assert "receipts/" in joined
        assert "September coffee" not in joined


def test_edit_receipt_updates_row(isolated_db):
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "15-Aug",
            "description": "Lunch",
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "Alpha",
            "amount": 40,
            "image_bytes": b"photo",
            "image_mime": "image/jpeg",
        },
    )
    with TestClient(app) as client:
        page = client.get("/receipts/1/edit")
        assert page.status_code == 200
        assert "Update receipt" in page.text
        assert 'value="Lunch"' in page.text
        saved = client.post(
            "/receipts/1/update",
            data={
                "date": "2026-08-16",
                "description": "Dinner",
                "category": "Food & Beverages",
                "project_code": "P1",
                "project_name": "Alpha",
                "amount": "35",
            },
            follow_redirects=True,
        )
        assert saved.status_code == 200
        assert "Dinner" in saved.text
        assert "Lunch" not in saved.text


def test_receipt_list_shows_thumbnail(isolated_db):
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "15-Aug",
            "description": "Lunch",
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "Alpha",
            "amount": 40,
            "image_bytes": b"photo",
            "image_mime": "image/jpeg",
        },
    )
    with TestClient(app) as client:
        response = client.get("/")
        assert 'class="thumb"' in response.text
        assert "/receipts/1/image" in response.text
        image = client.get("/receipts/1/image")
        assert image.status_code == 200
        assert image.content == b"photo"


def test_credit_card_analyze_fills_form(isolated_db, monkeypatch):
    from app import main as main_mod

    monkeypatch.setattr(
        main_mod,
        "analyze_receipt_bytes",
        lambda _raw: {
            "date": "2026-08-15",
            "amount": 88.5,
            "description": "ADNOC fuel",
            "category": "Parking",
        },
    )
    with TestClient(app) as client:
        response = client.post(
            "/credit-card/analyze",
            files={"photo": ("bill.jpg", b"fake-image", "image/jpeg")},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert "ADNOC fuel" in response.text
        assert "88.5" in response.text or 'value="88.5"' in response.text


def test_project_names_shown_in_datalist(isolated_db):
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "15-Aug",
            "description": "Lunch",
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "adnoc",
            "amount": 40,
        },
    )
    with TestClient(app) as client:
        response = client.get("/")
        assert 'list="names"' in response.text
        assert 'id="names"' in response.text
        assert "adnoc" in response.text


def test_download_all_button_names_the_month_being_viewed(isolated_db):
    with TestClient(app) as client:
        client.post(
            "/settings",
            data={
                "report_month_month": "Aug",
                "report_month_year": "2026",
                "month_view": "CURRENT",
            },
        )
        response = client.get("/")
        assert "Download Aug 2026" in response.text


def test_manage_page_shows_working_month_counts(isolated_db):
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "Coffee",
            "category": "Food & Beverages",
            "project_name": "mees",
            "amount": 12.5,
            "image_bytes": b"jpeg",
            "image_mime": "image/jpeg",
        },
    )
    with TestClient(app) as client:
        home = client.get("/")
        assert 'href="/manage"' in home.text
        response = client.get("/manage")
        assert response.status_code == 200
        assert "Manage" in response.text
        assert "Working month" in response.text
        assert "1 receipt" in response.text
        assert "12.50" in response.text
        assert "Clear working month" in response.text
        assert "Start new month" in response.text
        assert "SQLite backup" in response.text


def test_clear_current_requires_confirm_phrase(isolated_db):
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "Coffee",
            "category": "Food & Beverages",
            "project_name": "mees",
            "amount": 12.5,
        },
    )
    with TestClient(app) as client:
        refused = client.post(
            "/manage/clear-current",
            data={"confirm": "nope"},
            follow_redirects=True,
        )
        assert refused.status_code == 200
        assert "Type CLEAR" in refused.text
        from app.db import list_receipts

        assert len(list_receipts(isolated_db, None)) == 1
        cleared = client.post(
            "/manage/clear-current",
            data={"confirm": "CLEAR"},
            follow_redirects=True,
        )
        assert cleared.status_code == 200
        assert "Working month cleared" in cleared.text
        assert list_receipts(isolated_db, None) == []


def test_delete_archive_from_manage(isolated_db):
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "15-Aug",
            "description": "August lunch",
            "category": "Food & Beverages",
            "project_name": "mees",
            "amount": 16.0,
            "month_slug": "Aug_2026",
        },
    )
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "September coffee",
            "category": "Food & Beverages",
            "project_name": "mees",
            "amount": 12.0,
        },
    )
    with TestClient(app) as client:
        page = client.get("/manage")
        assert "Aug 2026" in page.text
        refused = client.post(
            "/manage/delete-archive",
            data={"month_slug": "Aug_2026", "confirm": "no"},
            follow_redirects=True,
        )
        assert "Type DELETE" in refused.text
        deleted = client.post(
            "/manage/delete-archive",
            data={"month_slug": "Aug_2026", "confirm": "DELETE"},
            follow_redirects=True,
        )
        assert "Deleted archive Aug 2026" in deleted.text
        from app.db import list_archived_months, list_receipts

        assert list_archived_months(isolated_db) == []
        assert list_receipts(isolated_db, None)[0]["description"] == "September coffee"


def test_restore_sqlite_from_manage(isolated_db, tmp_path):
    from app.db import init_db, list_receipts

    src = tmp_path / "restore.db"
    init_db(src)
    add_receipt(
        src,
        {
            "ref": 1,
            "date": "10-Sep",
            "description": "Restored lunch",
            "category": "Food & Beverages",
            "project_name": "mees",
            "amount": 40.0,
        },
    )
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "Will be replaced",
            "category": "Food & Beverages",
            "project_name": "mees",
            "amount": 9.0,
        },
    )
    with TestClient(app) as client:
        refused = client.post(
            "/manage/restore-backup",
            data={"confirm": "no"},
            files={"backup": ("restore.db", src.read_bytes(), "application/octet-stream")},
            follow_redirects=True,
        )
        assert "Type RESTORE" in refused.text
        assert list_receipts(isolated_db, None)[0]["description"] == "Will be replaced"
        restored = client.post(
            "/manage/restore-backup",
            data={"confirm": "RESTORE"},
            files={"backup": ("restore.db", src.read_bytes(), "application/octet-stream")},
            follow_redirects=True,
        )
        assert "Restored" in restored.text
        rows = list_receipts(isolated_db, None)
        assert len(rows) == 1
        assert rows[0]["description"] == "Restored lunch"


def test_clear_drafts_from_manage(isolated_db):
    from app.db import upsert_draft, working_month_stats

    upsert_draft(isolated_db, "99", "{}")
    with TestClient(app) as client:
        response = client.post("/manage/clear-drafts", follow_redirects=True)
        assert response.status_code == 200
        assert "Cleared 1 Telegram draft" in response.text
        assert working_month_stats(isolated_db)["drafts"] == 0

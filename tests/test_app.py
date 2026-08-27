from fastapi.testclient import TestClient

from app.db import add_credit_card, add_receipt, add_transport
from app.main import app, next_report_month


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["ok"] is True


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

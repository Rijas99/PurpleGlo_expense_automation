from fastapi.testclient import TestClient

from app import config
from app.db import (
    add_colleague,
    add_receipt,
    authenticate_user,
    get_user_by_telegram,
    init_db,
    link_telegram_invite,
    list_receipts,
    list_users,
)
from app.main import app
from app.telegram import parse_join_command


def test_init_db_creates_admin_user(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "admin_password", "test-admin")
    monkeypatch.setattr(config.settings, "employee_name", "Rijas Ali")
    db_path = tmp_path / "u.db"
    init_db(db_path)
    users = list_users(db_path)
    assert len(users) == 1
    assert users[0]["role"] == "admin"
    assert users[0]["username"] == "rijas"
    assert users[0]["name"] == "Rijas Ali"
    assert authenticate_user(db_path, "rijas", "test-admin")["id"] == users[0]["id"]
    assert authenticate_user(db_path, "rijas", "wrong") is None


def test_colleague_receipts_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "admin_password", "test-admin")
    db_path = tmp_path / "iso.db"
    init_db(db_path)
    admin = list_users(db_path)[0]
    colleague = add_colleague(db_path, name="Ahmed Khan", password="secret99")
    add_receipt(
        db_path,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "Admin lunch",
            "category": "Food & Beverages",
            "project_name": "mees",
            "amount": 16.0,
            "owner_id": admin["id"],
        },
    )
    add_receipt(
        db_path,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "Ahmed coffee",
            "category": "Food & Beverages",
            "project_name": "adnoc",
            "amount": 12.0,
            "owner_id": colleague["id"],
        },
    )
    mine = list_receipts(db_path, None, owner_id=admin["id"])
    his = list_receipts(db_path, None, owner_id=colleague["id"])
    assert [r["description"] for r in mine] == ["Admin lunch"]
    assert [r["description"] for r in his] == ["Ahmed coffee"]


def test_join_command_parses_invite_code():
    assert parse_join_command("join ABC123") == "ABC123"
    assert parse_join_command("JOIN abc123") == "ABC123"
    assert parse_join_command("adnoc") is None


def test_join_invite_links_telegram(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "admin_password", "test-admin")
    db_path = tmp_path / "join.db"
    init_db(db_path)
    colleague = add_colleague(db_path, name="Ahmed Khan", password="secret99")
    assert link_telegram_invite(db_path, colleague["invite_code"], "555001") is True
    linked = get_user_by_telegram(db_path, "555001")
    assert linked["id"] == colleague["id"]
    assert link_telegram_invite(db_path, "NOPE", "555001") is False


def test_admin_adds_colleague_from_manage(isolated_db, monkeypatch):
    monkeypatch.setattr(config.settings, "app_password", "on")
    with TestClient(app) as client:
        denied = client.post(
            "/manage/colleagues",
            data={"name": "Ahmed Khan", "password": "secret99"},
            follow_redirects=False,
        )
        assert denied.status_code == 303
        assert "/login" in denied.headers["location"]
        client.post("/login", data={"username": "rijas", "password": "test-admin"})
        added = client.post(
            "/manage/colleagues",
            data={"name": "Ahmed Khan", "password": "secret99"},
            follow_redirects=True,
        )
        assert added.status_code == 200
        assert "Ahmed Khan" in added.text
        assert "join" in added.text.lower()
        people = list_users(isolated_db)
        assert len(people) == 2
        assert {u["name"] for u in people} == {"Rijas Ali", "Ahmed Khan"}


def test_colleague_login_hides_admin_receipts(isolated_db, monkeypatch):
    monkeypatch.setattr(config.settings, "app_password", "on")
    admin = [u for u in list_users(isolated_db) if u["role"] == "admin"][0]
    colleague = add_colleague(isolated_db, name="Ahmed Khan", password="secret99")
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "Admin lunch",
            "category": "Food & Beverages",
            "project_name": "mees",
            "amount": 16.0,
            "owner_id": admin["id"],
        },
    )
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "Ahmed coffee",
            "category": "Food & Beverages",
            "project_name": "adnoc",
            "amount": 12.0,
            "owner_id": colleague["id"],
        },
    )
    with TestClient(app) as client:
        client.post("/login", data={"username": colleague["username"], "password": "secret99"})
        page = client.get("/")
        assert page.status_code == 200
        assert "Ahmed coffee" in page.text
        assert "Admin lunch" not in page.text
        manage = client.get("/manage")
        assert "Add colleague" not in manage.text


def test_member_cannot_add_colleague(isolated_db, monkeypatch):
    monkeypatch.setattr(config.settings, "app_password", "on")
    colleague = add_colleague(isolated_db, name="Ahmed Khan", password="secret99")
    with TestClient(app) as client:
        client.post("/login", data={"username": colleague["username"], "password": "secret99"})
        response = client.post(
            "/manage/colleagues",
            data={"name": "Someone Else", "password": "x"},
            follow_redirects=True,
        )
        assert "Only admin" in response.text
        assert len(list_users(isolated_db)) == 2


def test_first_telegram_user_auto_links_admin(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "admin_password", "test-admin")
    db_path = tmp_path / "tg.db"
    init_db(db_path)
    from app.db import auto_link_first_telegram, get_user_by_telegram

    linked = auto_link_first_telegram(db_path, "999111")
    assert linked["role"] == "admin"
    assert get_user_by_telegram(db_path, "999111")["id"] == linked["id"]
    colleague = add_colleague(db_path, name="Ahmed Khan", password="secret99")
    assert auto_link_first_telegram(db_path, "222333") is None
    assert get_user_by_telegram(db_path, "222333") is None
    assert colleague["telegram_user_id"] is None


def test_clear_working_month_only_selected_person(isolated_db, monkeypatch):
    monkeypatch.setattr(config.settings, "app_password", "on")
    admin = [u for u in list_users(isolated_db) if u["role"] == "admin"][0]
    colleague = add_colleague(isolated_db, name="Ahmed Khan", password="secret99")
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "Admin lunch",
            "category": "Food & Beverages",
            "project_name": "mees",
            "amount": 16.0,
            "owner_id": admin["id"],
        },
    )
    add_receipt(
        isolated_db,
        {
            "ref": 1,
            "date": "01-Sep",
            "description": "Ahmed coffee",
            "category": "Food & Beverages",
            "project_name": "adnoc",
            "amount": 12.0,
            "owner_id": colleague["id"],
        },
    )
    with TestClient(app) as client:
        client.post("/login", data={"username": "rijas", "password": "test-admin"})
        page = client.get("/manage")
        assert "Deletes only" in page.text
        assert "Rijas Ali" in page.text
        assert "Other people are not touched" in page.text
        cleared = client.post(
            "/manage/clear-current",
            data={"confirm": "CLEAR"},
            follow_redirects=True,
        )
        assert "Rijas Ali" in cleared.text
        assert "Other people were not touched" in cleared.text
        client.post("/manage/view-user", data={"user_id": colleague["id"]})
        other = client.get("/manage")
        assert "Deletes only" in other.text
        assert "Ahmed Khan" in other.text
    assert list_receipts(isolated_db, None, owner_id=admin["id"]) == []
    leftover = list_receipts(isolated_db, None, owner_id=colleague["id"])
    assert [r["description"] for r in leftover] == ["Ahmed coffee"]

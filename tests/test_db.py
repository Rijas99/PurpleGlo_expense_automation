from app.db import (
    add_credit_card,
    add_receipt,
    add_transport,
    delete_receipt,
    init_db,
    list_credit_card,
    list_receipts,
    list_transport,
    next_receipt_ref,
    project_names,
    update_credit_card,
    update_receipt,
    update_transport,
)


def test_receipt_crud_roundtrip(tmp_path):
    db_path = tmp_path / "t.db"
    init_db(db_path)
    ref = next_receipt_ref(db_path)
    assert ref == 1
    add_receipt(
        db_path,
        {
            "ref": ref,
            "date": "15-Aug",
            "description": "Lunch",
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "Alpha",
            "amount": 40.0,
            "image_bytes": b"fake-jpeg",
            "image_mime": "image/jpeg",
        },
    )
    rows = list_receipts(db_path, month_slug=None)
    assert len(rows) == 1
    assert rows[0]["description"] == "Lunch"
    assert rows[0]["image_bytes"] == b"fake-jpeg"
    assert next_receipt_ref(db_path) == 2
    ok = delete_receipt(db_path, ref=1)
    assert ok is True
    assert list_receipts(db_path, month_slug=None) == []


def _add(db_path, ref: int, description: str) -> None:
    add_receipt(
        db_path,
        {
            "ref": ref,
            "date": "15-Aug",
            "description": description,
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "Alpha",
            "amount": 10.0,
            "image_bytes": description.encode(),
            "image_mime": "image/jpeg",
        },
    )


def test_delete_receipt_renumbers_remaining_refs(tmp_path):
    db_path = tmp_path / "renumber.db"
    init_db(db_path)
    _add(db_path, 1, "one")
    _add(db_path, 2, "two")
    _add(db_path, 3, "three")
    _add(db_path, 4, "four")

    assert delete_receipt(db_path, ref=3) is True
    rows = list_receipts(db_path, month_slug=None)
    assert [r["ref"] for r in rows] == [1, 2, 3]
    assert [r["description"] for r in rows] == ["one", "two", "four"]
    assert rows[2]["image_bytes"] == b"four"
    assert next_receipt_ref(db_path) == 4


def test_update_receipt_keeps_photo_unless_replaced(tmp_path):
    db_path = tmp_path / "edit.db"
    init_db(db_path)
    _add(db_path, 1, "Lunch")
    assert update_receipt(
        db_path,
        1,
        {
            "date": "16-Aug",
            "description": "Dinner",
            "category": "Food & Beverages",
            "project_code": "P2",
            "project_name": "Beta",
            "amount": 22.0,
        },
    )
    row = list_receipts(db_path, None)[0]
    assert row["description"] == "Dinner"
    assert row["project_name"] == "Beta"
    assert row["amount"] == 22.0
    assert row["image_bytes"] == b"Lunch"
    assert update_receipt(
        db_path,
        1,
        {
            "date": "16-Aug",
            "description": "Dinner",
            "category": "Food & Beverages",
            "project_code": "P2",
            "project_name": "Beta",
            "amount": 22.0,
            "image_bytes": b"new-photo",
            "image_mime": "image/jpeg",
        },
        replace_image=True,
    )
    assert list_receipts(db_path, None)[0]["image_bytes"] == b"new-photo"


def test_update_credit_card_and_transport(tmp_path):
    db_path = tmp_path / "edit2.db"
    init_db(db_path)
    add_credit_card(
        db_path,
        {
            "date": "15-Aug",
            "description": "AWS",
            "category": "Subscriptions",
            "project_code": "",
            "project_name": "adnoc",
            "amount": 20,
            "image_bytes": b"cc",
            "image_mime": "image/jpeg",
        },
    )
    cc_id = list_credit_card(db_path, None)[0]["id"]
    assert update_credit_card(
        db_path,
        cc_id,
        {
            "date": "16-Aug",
            "description": "Azure",
            "category": "Subscriptions",
            "project_code": "P1",
            "project_name": "adnoc",
            "amount": 30,
        },
    )
    assert list_credit_card(db_path, None)[0]["description"] == "Azure"
    add_transport(
        db_path,
        {
            "date": "17-Aug",
            "from_location": "Dubai",
            "destination": "Abu Dhabi",
            "return_included": True,
            "project_code": "",
            "project_name": "adnoc",
        },
    )
    tr_id = list_transport(db_path, None)[0]["id"]
    assert update_transport(
        db_path,
        tr_id,
        {
            "date": "18-Aug",
            "from_location": "Sharjah",
            "destination": "Dubai",
            "return_included": False,
            "project_code": "P9",
            "project_name": "site",
        },
    )
    row = list_transport(db_path, None)[0]
    assert row["from_location"] == "Sharjah"
    assert row["return_included"] in (0, False)


def test_project_names_include_archived(tmp_path):
    db_path = tmp_path / "names.db"
    init_db(db_path)
    add_receipt(
        db_path,
        {
            "ref": 1,
            "date": "15-Aug",
            "description": "Lunch",
            "category": "Food & Beverages",
            "project_code": "P1",
            "project_name": "adnoc",
            "amount": 10,
            "month_slug": "Aug_2026",
        },
    )
    add_transport(
        db_path,
        {
            "date": "01-Sep",
            "from_location": "Dubai",
            "destination": "Abu Dhabi",
            "return_included": False,
            "project_name": "site alpha",
        },
    )
    names = project_names(db_path)
    assert "adnoc" in names
    assert "site alpha" in names

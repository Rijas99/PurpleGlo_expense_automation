from datetime import datetime, time as dt_time

from app.corrections import apply_corrections, get_meal_description, to_input_date
from app.constants import FOOD_CAP_AMOUNT


def test_meal_breakfast_lunch_dinner():
    assert get_meal_description(dt_time(5, 0)) == "Breakfast"
    assert get_meal_description(dt_time(8, 0)) == "Breakfast"
    assert get_meal_description(dt_time(11, 0)) == "Breakfast"
    assert get_meal_description(dt_time(11, 15)) == "Lunch"
    assert get_meal_description(dt_time(13, 0)) == "Lunch"
    assert get_meal_description(dt_time(18, 0)) == "Lunch"
    assert get_meal_description(dt_time(18, 15)) == "Dinner"
    assert get_meal_description(dt_time(20, 0)) == "Dinner"
    assert get_meal_description(dt_time(3, 0)) == "Dinner"


def test_correction_updates_amount():
    draft = {"amount": 10.0, "description": "Lunch", "category": "Food & Beverages"}
    updated = apply_corrections(draft, "amount 52")
    assert updated["amount"] == 52.0


def test_correction_matches_category_case_insensitive():
    draft = {"category": "Can't classify"}
    updated = apply_corrections(draft, "category parking")
    assert updated["category"] == "Parking"


def test_correction_project_code_and_name():
    draft = {"project_code": "", "project_name": ""}
    updated = apply_corrections(draft, "project 250909-PDS-303\nproject name Site Alpha")
    assert updated["project_code"] == "250909-PDS-303"
    assert updated["project_name"] == "Site Alpha"


def test_caption_project_name_with_colon_spaces():
    draft = {"project_code": "", "project_name": ""}
    updated = apply_corrections(draft, "project name : adnoc")
    assert updated["project_name"] == "adnoc"


def test_photo_caption_plain_text_sets_project_name():
    from app.bot import build_receipt_draft

    draft = build_receipt_draft(
        {
            "date": "2026-08-15",
            "time": "13:00",
            "amount": 50,
            "description": "Lunch",
            "category": "Food & Beverages",
        },
        caption="adnoc",
    )
    assert draft["kind"] == "receipt"
    assert draft["project_name"] == "adnoc"
    assert draft["description"] == "Lunch"


def test_photo_caption_dis_adds_note_to_food_description():
    from app.bot import build_receipt_draft

    draft = build_receipt_draft(
        {
            "date": "2026-08-15",
            "time": "13:00",
            "amount": 45,
            "description": "Restaurant meal",
            "category": "Food & Beverages",
        },
        caption="adnoc\ndis: ali and rijas\ncap,40",
    )
    assert draft["project_name"] == "adnoc"
    assert draft["description"].startswith("Lunch, ali and rijas")
    assert draft["amount"] == 40.0
    assert "capped at 40" in draft["description"]


def test_photo_caption_cc_sets_credit_card_kind():
    from app.bot import build_receipt_draft

    draft = build_receipt_draft(
        {
            "date": "2026-08-15",
            "amount": 50,
            "description": "Lunch",
            "category": "Food & Beverages",
        },
        caption="CC, adnoc",
    )
    assert draft["kind"] == "credit_card"
    assert draft["project_name"] == "adnoc"



def test_correction_date_iso_and_description():
    draft = {"date": "2026-01-01", "description": "x"}
    updated = apply_corrections(draft, "date 2026-08-15\ndescription Dinner")
    assert updated["date"] == "2026-08-15"
    assert updated["description"] == "Dinner"


def test_cap_command_reduces_food_amount_and_notes_description():
    draft = {
        "amount": 75.0,
        "description": "Lunch",
        "category": "Food & Beverages",
        "time": "13:00",
    }
    updated = apply_corrections(draft, "cap")
    assert updated["amount"] == FOOD_CAP_AMOUNT
    assert "capped at 40" in updated["description"]


def test_cap_comma_amount_uses_given_limit():
    draft = {
        "amount": 45.0,
        "description": "Lunch",
        "category": "Food & Beverages",
        "time": "13:00",
    }
    updated = apply_corrections(draft, "cap,40")
    assert updated["amount"] == 40.0
    assert "capped at 40" in updated["description"]
    updated = apply_corrections(draft, "cap, 30")
    assert updated["amount"] == 30.0
    assert "capped at 30" in updated["description"]


def test_dis_appends_to_food_meal_name():
    draft = {
        "amount": 45.0,
        "description": "Lunch",
        "category": "Food & Beverages",
        "time": "13:00",
        "project_name": "adnoc",
    }
    updated = apply_corrections(draft, "dis: ali and rijas")
    assert updated["description"] == "Lunch, ali and rijas"
    assert updated["project_name"] == "adnoc"


def test_corrections_do_not_mutate_original():
    draft = {"amount": 1.0}
    apply_corrections(draft, "amount 9")
    assert draft["amount"] == 1.0


def test_unknown_text_becomes_project_name():
    draft = {"amount": 3.0, "description": "Parking", "project_name": ""}
    updated = apply_corrections(draft, "adnoc")
    assert updated["amount"] == 3.0
    assert updated["description"] == "Parking"
    assert updated["project_name"] == "adnoc"


def test_to_input_date_from_display():
    assert to_input_date("15-Aug", year=2026) == "2026-08-15"
    assert to_input_date("2026-08-15") == "2026-08-15"

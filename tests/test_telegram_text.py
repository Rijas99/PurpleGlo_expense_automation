from app.telegram import (
    format_draft_message,
    is_start_command,
    parse_save_command,
    parse_telegram_note,
)


def test_format_draft_includes_core_fields():
    text = format_draft_message(
        {
            "date": "2026-08-15",
            "time": "13:00",
            "amount": 42.5,
            "description": "Lunch",
            "category": "Food & Beverages",
            "project_code": "",
            "project_name": "",
        }
    )
    assert "Lunch" in text
    assert "42.50" in text
    assert "Food & Beverages" in text
    assert "save / ok" in text


def test_parse_save_command():
    assert parse_save_command("save") is True
    assert parse_save_command("OK") is True
    assert parse_save_command("yes") is True
    assert parse_save_command("amount 10") is False


def test_start_command_matches_bot_suffix():
    assert is_start_command("/start") is True
    assert is_start_command("/start@purpleglo_expense_bot") is True
    assert is_start_command("amount 10") is False


def test_greetings_ask_for_commands():
    for text in ("hi", "Hi", "hy", "hey", "hello", "hello!", "good morning", "help"):
        assert is_start_command(text) is True
    assert is_start_command("adnoc") is False
    assert is_start_command("TR, Dubai, Abu Dhabi, adnoc") is False


def test_caption_plain_text_is_project_name():
    note = parse_telegram_note("adnoc")
    assert note["kind"] == "receipt"
    assert note["project_name"] == "adnoc"


def test_caption_project_and_dis():
    note = parse_telegram_note("adnoc\ndis: ali and rijas")
    assert note["kind"] == "receipt"
    assert note["project_name"] == "adnoc"
    assert note["extra_description"] == "ali and rijas"
    note = parse_telegram_note("adnoc dis: ali and rijas")
    assert note["project_name"] == "adnoc"
    assert note["extra_description"] == "ali and rijas"


def test_caption_includes_cap_amount():
    note = parse_telegram_note("adnoc\ncap,40")
    assert note["project_name"] == "adnoc"
    assert note["cap_amount"] == 40.0


def test_caption_cc_prefix_is_credit_card():
    note = parse_telegram_note("CC, adnoc")
    assert note["kind"] == "credit_card"
    assert note["project_name"] == "adnoc"
    note = parse_telegram_note("cc: Site Alpha")
    assert note["kind"] == "credit_card"
    assert note["project_name"] == "Site Alpha"


def test_caption_tr_is_transport():
    note = parse_telegram_note("TR, Dubai, Abu Dhabi, adnoc")
    assert note["kind"] == "transport"
    assert note["from_location"] == "Dubai"
    assert note["destination"] == "Abu Dhabi"
    assert note["project_name"] == "adnoc"
    assert note["return_included"] is False


def test_caption_tr_with_return():
    note = parse_telegram_note("TR, Dubai, Abu Dhabi, adnoc, return")
    assert note["kind"] == "transport"
    assert note["return_included"] is True
    assert note["project_name"] == "adnoc"

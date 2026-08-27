# PurpleGlo Expense Manager — FastAPI + Telegram + Excel template

Date: 2026-08-27

## Problem

Streamlit + Supabase is slow after idle (cloud cold start) and export does not match the company expense form. Receipt capture should work from the web and from Telegram, with a correction loop before save.

## Stack

- FastAPI + Jinja2 HTML (replace Streamlit)
- SQLite (replace Supabase); receipt images stored as BLOBs so one DB file is a full backup
- Gemini 2.5 Flash for receipt OCR (unchanged)
- Telegram Bot API webhook (and polling for local)
- `openpyxl` fills `excel format/Expense Form.xlsx` in place (logo, layout, TOTAL formula kept)

## Capture paths

1. Web: upload photo → Gemini → editable form → save
2. Telegram: send photo → bot replies with extracted fields → user sends corrections in text → `save` / tap Save → stored

Both write the same SQLite database. Credit card and transport stay as web (and Telegram text commands later if needed); v1 Telegram is receipts-first, with web covering all three tabs.

## Excel mapping (Sheet1)

| Cell / range | Field |
|---|---|
| D9 | Employee name (default Rijas Ali) |
| D10 | Report month (e.g. Aug 2026) |
| C13:H33 | REF, DATE, DESCRIPTION, CATEGORY, PROJECT CODE, AMOUNT |
| H34 | `=SUM(H13:H33)` — keep / extend if extra rows inserted |

Project Name is stored in SQLite only (not on this form). Categories match Sheet2.

## Free cloud

Docker, `PORT`, `/health`. First request after sleep is faster than Streamlit. SQLite on ephemeral disk can reset on redeploy — UI offers DB backup download. Optional keep-alive against `/health`.

## Auth

- Web: shared `APP_PASSWORD`
- Telegram: `TELEGRAM_ALLOWED_USER_IDS` allowlist

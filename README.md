# PurpleGlo Expense Manager

FastAPI app: web upload + Telegram receipts + fill the company **Employee Expense Form**.

Replaces Streamlit and Supabase. Data lives in SQLite (`data/expenses.db`), including receipt photos.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your Gemini key in `.env` as `GOOGLE_API_KEY`. Then:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

## Telegram

1. In Telegram, talk to [@BotFather](https://t.me/BotFather), create a bot, copy the token into `TELEGRAM_BOT_TOKEN`.
2. Optionally set `TELEGRAM_ALLOWED_USER_IDS` to your numeric user id (get it from `@userinfobot`).
3. Locally, leave `TELEGRAM_WEBHOOK_URL` empty — the app **polls** Telegram while it is running.
4. Send a receipt **photo** to the bot. It replies with date, amount, category, etc.
5. Correct in plain text, one per line:

```
amount 52
category Parking
description Lunch
date 2026-08-15
project 250909-PDS-303
project name Site Alpha
cap
```

6. Send `save` / `ok` or tap **Save receipt**. Project name is required.

## Excel export

**Download expense form + bills** copies `excel format/Expense Form.xlsx`, writes NAME, MONTH, and rows (REF, DATE, DESCRIPTION, CATEGORY, PROJECT CODE, AMOUNT), keeps the TOTAL formula and logo, and zips receipt images as `1.jpg`, `2.jpg`, …

## Free cloud (Render)

This app can run on Render’s [free web service](https://render.com/docs/your-first-deploy). Free instances sleep after 15 minutes of no traffic, and the disk is wiped on restart — download a SQLite backup often.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Rijas99/PurpleGlo_expense_automation)

1. Sign up at Render with GitHub (free, no card).
2. Click the button above, or **New → Web Service** and select this repo (Docker, free plan).
3. When prompted, paste `GOOGLE_API_KEY`, `TELEGRAM_BOT_TOKEN`, `APP_PASSWORD`, and set `TELEGRAM_WEBHOOK_URL` to `https://YOUR-SERVICE.onrender.com`.
4. After deploy, stop any local `uvicorn` so Telegram is not polled twice.

## Tests

```bash
python -m pytest tests -v
```

The old Streamlit app remains in `expense_app.py` as a reference and is not used.

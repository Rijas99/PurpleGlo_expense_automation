# FastAPI Telegram Excel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline; user said start).

**Goal:** Replace Streamlit/Supabase with FastAPI + SQLite + Telegram + fill the existing Expense Form.xlsx.

**Architecture:** One FastAPI process serves HTML pages, REST save/list/delete, Gemini analyze, Telegram webhook, and Excel/zip export. Domain logic (corrections, meal labels, Excel fill, SQLite CRUD) lives in importable modules with pytest coverage.

**Tech Stack:** FastAPI, Jinja2, SQLite, Pillow, google-generativeai, openpyxl, httpx, python-dotenv, pytest.

## Global Constraints

- Do not rewrite the Excel layout; copy the template then write cells.
- Keep Gemini JSON keys: date, time, amount, description, category.
- Categories stay the Sheet2 list including `Can't classify`.
- Food cap at 40 SAR/AED when requested; meal name from time for Food & Beverages.
- No Streamlit, no Supabase in the new app.
- Do not commit secrets. User did not request git commits during implementation.

---

### Task 1: Domain helpers + Excel fill (TDD)

**Files:**
- Create: `app/constants.py`, `app/corrections.py`, `app/excel_fill.py`, `app/db.py`
- Test: `tests/test_corrections.py`, `tests/test_excel_fill.py`, `tests/test_db.py`

- [ ] Tests then implementation for correction parsing, meal description, Excel cell mapping, SQLite CRUD.

### Task 2: FastAPI web app

**Files:**
- Create: `app/config.py`, `app/gemini.py`, `app/main.py`, `templates/*.html`, `static/style.css`

- [ ] Receipts, credit card, transport pages; upload+analyze; save/delete; month archive; password gate.

### Task 3: Telegram receipts

**Files:**
- Create: `app/telegram.py`

- [ ] Photo → Gemini → reply; text corrections; save; allowlist.

### Task 4: Export + deploy files

**Files:**
- Create: `app/export.py`, `Dockerfile`, `.env.example`, `requirements.txt`, `README.md`

- [ ] Fill Expense Form + bills zip; health; backup download.

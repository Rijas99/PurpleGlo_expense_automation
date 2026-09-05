from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class Settings:
    google_api_key: str = _env("GOOGLE_API_KEY")
    telegram_bot_token: str = _env("TELEGRAM_BOT_TOKEN")
    telegram_webhook_url: str = _env("TELEGRAM_WEBHOOK_URL")
    telegram_webhook_secret: str = _env("TELEGRAM_WEBHOOK_SECRET", "purpleglo-hook")
    telegram_allowed_user_ids: str = _env("TELEGRAM_ALLOWED_USER_IDS")
    app_password: str = _env("APP_PASSWORD")
    secret_key: str = _env("SECRET_KEY", "purpleglo-dev-secret-change-me")
    employee_name: str = _env("EMPLOYEE_NAME", "Rijas Ali")
    database_path: Path = Path(_env("DATABASE_PATH") or str(ROOT / "data" / "expenses.db"))
    template_path: Path = ROOT / "excel format" / "Expense Form.xlsx"
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")
    turso_database_url: str = _env("TURSO_DATABASE_URL")
    turso_auth_token: str = _env("TURSO_AUTH_TOKEN")

    def allowed_telegram_ids(self) -> set[str]:
        if not self.telegram_allowed_user_ids:
            return set()
        return {x.strip() for x in self.telegram_allowed_user_ids.split(",") if x.strip()}


settings = Settings()

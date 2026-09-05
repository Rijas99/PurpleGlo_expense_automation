from __future__ import annotations

import pytest

from app import config
from app.db import init_db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    path = tmp_path / "expenses.db"
    monkeypatch.setattr(config.settings, "database_path", path)
    monkeypatch.setattr(config.settings, "app_password", "")
    monkeypatch.setattr(config.settings, "telegram_bot_token", "")
    monkeypatch.setattr(config.settings, "telegram_webhook_url", "")
    monkeypatch.setattr(config.settings, "turso_database_url", "")
    monkeypatch.setattr(config.settings, "turso_auth_token", "")
    init_db(path)
    return path

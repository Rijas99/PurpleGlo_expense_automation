from pathlib import Path

from app import config
from app.backup import restore_sqlite_backup, upload_sqlite_backup
from app.db import add_receipt, init_db, list_receipts


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, headers=None):
        self.calls.append(("GET", url))
        return self.responses.pop(0)

    def put(self, url, headers=None, json=None):
        self.calls.append(("PUT", url, json))
        return self.responses.pop(0)


def test_restore_writes_sqlite_file(tmp_path, monkeypatch):
    import base64

    monkeypatch.setattr(config.settings, "github_backup_repo", "Rijas99/PurpleGlo_expense_data")
    monkeypatch.setattr(config.settings, "github_backup_token", "test-token")
    dest = tmp_path / "expenses.db"
    encoded = base64.b64encode(b"sqlite-bytes").decode("ascii")
    fake = _FakeClient([_FakeResponse(200, {"content": encoded, "sha": "abc"})])
    monkeypatch.setattr("app.backup.httpx.Client", lambda timeout=None: fake)
    assert restore_sqlite_backup(dest) is True
    assert dest.read_bytes() == b"sqlite-bytes"


def test_upload_sends_existing_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "github_backup_repo", "Rijas99/PurpleGlo_expense_data")
    monkeypatch.setattr(config.settings, "github_backup_token", "test-token")
    dest = tmp_path / "expenses.db"
    dest.write_bytes(b"x" * 200)
    fake = _FakeClient(
        [
            _FakeResponse(200, {"sha": "old-sha"}),
            _FakeResponse(200, {"sha": "new-sha"}),
        ]
    )
    monkeypatch.setattr("app.backup.httpx.Client", lambda timeout=None: fake)
    upload_sqlite_backup(dest)
    assert fake.calls[1][0] == "PUT"
    assert fake.calls[1][2]["sha"] == "old-sha"


def test_add_receipt_uploads_when_backup_enabled(tmp_path, monkeypatch):
    db_path = tmp_path / "expenses.db"
    init_db(db_path)
    monkeypatch.setattr(config.settings, "github_backup_repo", "Rijas99/PurpleGlo_expense_data")
    monkeypatch.setattr(config.settings, "github_backup_token", "test-token")
    uploaded = []

    def fake_upload(path):
        uploaded.append(Path(path))

    monkeypatch.setattr("app.db.upload_sqlite_backup", fake_upload)
    add_receipt(
        db_path,
        {
            "ref": 1,
            "date": "15-Aug",
            "description": "Lunch",
            "category": "Food & Beverages",
            "project_name": "mees",
            "amount": 16.0,
            "image_bytes": b"jpeg",
            "image_mime": "image/jpeg",
        },
    )
    assert uploaded == [db_path]
    assert list_receipts(db_path, None)[0]["description"] == "Lunch"

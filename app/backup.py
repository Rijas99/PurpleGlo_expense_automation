from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

from app.config import settings

log = logging.getLogger(__name__)
_FILE_NAME = "expenses.db"


def github_backup_enabled() -> bool:
    return bool(settings.github_backup_repo and settings.github_backup_token)


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.github_backup_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _contents_url() -> str:
    repo = settings.github_backup_repo.strip().strip("/")
    return f"https://api.github.com/repos/{repo}/contents/{_FILE_NAME}"


def restore_sqlite_backup(db_path: Path | str) -> bool:
    """Download the remote SQLite file onto disk. Returns True if a backup was restored."""
    if not github_backup_enabled():
        return False
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(_contents_url(), headers=_headers())
    if resp.status_code == 404:
        log.info("No GitHub SQLite backup yet")
        return False
    if resp.status_code != 200:
        raise RuntimeError(f"Could not download expense backup ({resp.status_code})")
    payload = resp.json()
    encoded = str(payload.get("content") or "").replace("\n", "")
    if not encoded:
        raise RuntimeError("GitHub backup file was empty")
    path.write_bytes(base64.b64decode(encoded))
    log.info("Restored SQLite backup from GitHub")
    return True


def upload_sqlite_backup(db_path: Path | str) -> None:
    if not github_backup_enabled():
        return
    path = Path(db_path)
    if not path.exists() or path.stat().st_size < 100:
        return
    raw = path.read_bytes()
    with httpx.Client(timeout=60.0) as client:
        current = client.get(_contents_url(), headers=_headers())
        sha = None
        if current.status_code == 200:
            sha = current.json().get("sha")
        elif current.status_code not in {404, 200}:
            raise RuntimeError(f"Could not read expense backup ({current.status_code})")
        body = {
            "message": "Update expense database backup",
            "content": base64.b64encode(raw).decode("ascii"),
        }
        if sha:
            body["sha"] = sha
        put = client.put(_contents_url(), headers=_headers(), json=body)
    if put.status_code not in {200, 201}:
        raise RuntimeError(f"Could not upload expense backup ({put.status_code})")
    log.info("Uploaded SQLite backup to GitHub")

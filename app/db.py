from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.auth import hash_password, new_invite_code, unique_username, username_from_name, verify_password
from app.backup import github_backup_enabled, upload_sqlite_backup
from app.config import settings


class _Row:
    """sqlite3.Row-like mapping for libsql tuple rows."""

    def __init__(self, pairs: list[tuple[str, Any]]):
        self._data = dict(pairs)
        self._keys = [k for k, _ in pairs]
        self._vals = [v for _, v in pairs]

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._vals[key]
        return self._data[key]


class _CompatCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        names = [d[0] for d in (self._cursor.description or [])]
        if not names:
            return row
        return _Row(list(zip(names, row)))

    def fetchall(self):
        rows = self._cursor.fetchall()
        names = [d[0] for d in (self._cursor.description or [])]
        if not names:
            return list(rows)
        return [_Row(list(zip(names, row))) for row in rows]

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def description(self):
        return self._cursor.description

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _CompatConn:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=None):
        if params is None:
            return _CompatCursor(self._conn.execute(sql))
        return _CompatCursor(self._conn.execute(sql, params))

    def executescript(self, sql):
        return self._conn.executescript(sql)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _PersistingConn:
    def __init__(self, conn, db_path: Path):
        self._conn = conn
        self._db_path = db_path
        self._dirty = False

    def execute(self, sql, params=None):
        if params is None:
            return self._conn.execute(sql)
        return self._conn.execute(sql, params)

    def executescript(self, sql):
        return self._conn.executescript(sql)

    def commit(self):
        self._conn.commit()
        self._dirty = True

    def close(self):
        try:
            if self._dirty:
                upload_sqlite_backup(self._db_path)
        finally:
            self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def turso_enabled() -> bool:
    return bool(settings.turso_database_url and settings.turso_auth_token)


def _connect(db_path: Path | str, persist: bool = True):
    if turso_enabled():
        import libsql

        conn = libsql.connect(settings.turso_database_url, auth_token=settings.turso_auth_token)
        return _CompatConn(conn)
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    if persist and github_backup_enabled():
        return _PersistingConn(conn, path)
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    telegram_user_id TEXT,
    invite_code TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ref INTEGER NOT NULL,
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    project_code TEXT,
    project_name TEXT NOT NULL DEFAULT '',
    amount REAL NOT NULL,
    image_bytes BLOB,
    image_mime TEXT,
    month_slug TEXT,
    owner_id INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS credit_card (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ref INTEGER,
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    project_code TEXT,
    project_name TEXT NOT NULL DEFAULT '',
    amount REAL NOT NULL,
    image_bytes BLOB,
    image_mime TEXT,
    month_slug TEXT,
    owner_id INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS transport (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    from_location TEXT NOT NULL,
    destination TEXT NOT NULL,
    return_included INTEGER NOT NULL DEFAULT 0,
    project_code TEXT,
    project_name TEXT NOT NULL DEFAULT '',
    month_slug TEXT,
    owner_id INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS drafts (
    chat_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    image_bytes BLOB,
    image_mime TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_receipts_month ON receipts(month_slug);
CREATE INDEX IF NOT EXISTS idx_cc_month ON credit_card(month_slug);
CREATE INDEX IF NOT EXISTS idx_transport_month ON transport(month_slug);
"""


def init_db(db_path: Path | str) -> None:
    conn = _connect(db_path, persist=False)
    try:
        conn.executescript(_SCHEMA)
        _migrate_credit_card(conn)
        _migrate_owner_columns(conn)
        _seed_admin(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate_credit_card(conn) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(credit_card)").fetchall()}
    if "image_bytes" not in cols:
        conn.execute("ALTER TABLE credit_card ADD COLUMN image_bytes BLOB")
    if "image_mime" not in cols:
        conn.execute("ALTER TABLE credit_card ADD COLUMN image_mime TEXT")
    if "ref" not in cols:
        conn.execute("ALTER TABLE credit_card ADD COLUMN ref INTEGER")
    missing = conn.execute(
        "SELECT id FROM credit_card WHERE ref IS NULL ORDER BY id"
    ).fetchall()
    if not missing:
        return
    current = conn.execute(
        "SELECT id FROM credit_card WHERE month_slug IS NULL ORDER BY id"
    ).fetchall()
    for i, row in enumerate(current, start=1):
        conn.execute("UPDATE credit_card SET ref = ? WHERE id = ?", (i, row["id"]))
    archived = conn.execute(
        "SELECT DISTINCT month_slug FROM credit_card WHERE month_slug IS NOT NULL"
    ).fetchall()
    for slug_row in archived:
        slug = slug_row["month_slug"]
        rows = conn.execute(
            "SELECT id FROM credit_card WHERE month_slug = ? ORDER BY id",
            (slug,),
        ).fetchall()
        for i, row in enumerate(rows, start=1):
            conn.execute("UPDATE credit_card SET ref = ? WHERE id = ?", (i, row["id"]))


def _table_columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate_owner_columns(conn) -> None:
    for table in ("receipts", "credit_card", "transport"):
        cols = _table_columns(conn, table)
        if "owner_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN owner_id INTEGER NOT NULL DEFAULT 1")


def _seed_admin(conn) -> int:
    row = conn.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1").fetchone()
    if row:
        admin_id = int(row["id"])
    else:
        name = settings.employee_name or "Rijas Ali"
        password = settings.admin_password or settings.app_password or "Purpleglo@321"
        conn.execute(
            """
            INSERT INTO users (username, name, password_hash, role, invite_code)
            VALUES (?, ?, ?, 'admin', ?)
            """,
            (username_from_name(name), name, hash_password(password), new_invite_code()),
        )
        admin_id = int(
            conn.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1").fetchone()["id"]
        )
    conn.execute(
        "UPDATE receipts SET owner_id = ? WHERE owner_id IS NULL",
        (admin_id,),
    )
    conn.execute(
        "UPDATE credit_card SET owner_id = ? WHERE owner_id IS NULL",
        (admin_id,),
    )
    conn.execute(
        "UPDATE transport SET owner_id = ? WHERE owner_id IS NULL",
        (admin_id,),
    )
    linked = conn.execute(
        "SELECT telegram_user_id FROM users WHERE id = ?", (admin_id,)
    ).fetchone()
    if not (linked and linked["telegram_user_id"]):
        chats = conn.execute("SELECT DISTINCT chat_id FROM drafts").fetchall()
        if len(chats) == 1:
            conn.execute(
                "UPDATE users SET telegram_user_id = ? WHERE id = ?",
                (str(chats[0]["chat_id"]), admin_id),
            )
    return admin_id


def _public_user(row) -> dict[str, Any]:
    data = dict(row)
    data.pop("password_hash", None)
    return data


def _scope(month_slug: str | None, owner_id: int | None) -> tuple[str, list]:
    parts = []
    params: list = []
    if month_slug:
        parts.append("month_slug = ?")
        params.append(month_slug)
    else:
        parts.append("month_slug IS NULL")
    if owner_id is not None:
        parts.append("owner_id = ?")
        params.append(owner_id)
    return " AND ".join(parts), params


def _resolve_owner_id(conn, owner_id: int | None) -> int:
    if owner_id is not None:
        return int(owner_id)
    row = conn.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1").fetchone()
    if row:
        return int(row["id"])
    return 1


def list_users(db_path: Path | str) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, username, name, role, telegram_user_id, invite_code FROM users ORDER BY role, name"
        ).fetchall()
        return _rows(rows)
    finally:
        conn.close()


def get_user(db_path: Path | str, user_id: int) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, username, name, role, telegram_user_id, invite_code FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_telegram(db_path: Path | str, telegram_user_id: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, username, name, role, telegram_user_id, invite_code FROM users WHERE telegram_user_id = ?",
            (str(telegram_user_id),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def authenticate_user(db_path: Path | str, username: str, password: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE lower(username) = lower(?)",
            ((username or "").strip(),),
        ).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            return None
        return _public_user(row)
    finally:
        conn.close()


def add_colleague(db_path: Path | str, name: str, password: str) -> dict[str, Any]:
    clean = (name or "").strip()
    if not clean:
        raise RuntimeError("Name is required.")
    if not (password or "").strip():
        raise RuntimeError("Password is required.")
    conn = _connect(db_path)
    try:
        existing = {
            str(row["username"]).lower()
            for row in conn.execute("SELECT username FROM users").fetchall()
        }
        username = unique_username(existing, clean)
        invite = new_invite_code()
        conn.execute(
            """
            INSERT INTO users (username, name, password_hash, role, invite_code)
            VALUES (?, ?, ?, 'member', ?)
            """,
            (username, clean, hash_password(password), invite),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, username, name, role, telegram_user_id, invite_code FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def delete_colleague(db_path: Path | str, user_id: int) -> None:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            raise RuntimeError("Person not found.")
        if row["role"] == "admin":
            raise RuntimeError("Cannot delete the admin account.")
        for table in ("receipts", "credit_card", "transport"):
            conn.execute(f"DELETE FROM {table} WHERE owner_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def auto_link_first_telegram(db_path: Path | str, telegram_user_id: str) -> dict[str, Any] | None:
    existing = get_user_by_telegram(db_path, telegram_user_id)
    if existing:
        return existing
    conn = _connect(db_path)
    try:
        people = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        if len(people) != 1:
            return None
        admin = people[0]
        if admin["telegram_user_id"]:
            return None
        conn.execute(
            "UPDATE users SET telegram_user_id = ? WHERE id = ?",
            (str(telegram_user_id), admin["id"]),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (admin["id"],)).fetchone()
        return _public_user(row) if row else None
    finally:
        conn.close()


def link_telegram_invite(db_path: Path | str, invite_code: str, telegram_user_id: str) -> bool:
    code = (invite_code or "").strip().upper()
    if not code:
        return False
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id FROM users WHERE upper(invite_code) = ?", (code,)
        ).fetchone()
        if not row:
            return False
        conn.execute(
            "UPDATE users SET telegram_user_id = ? WHERE id = ?",
            (str(telegram_user_id), row["id"]),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def _rows(cursor_result) -> list[dict[str, Any]]:
    return [dict(r) for r in cursor_result]


def list_receipts(
    db_path: Path | str, month_slug: str | None, owner_id: int | None = None
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        where, params = _scope(month_slug, owner_id)
        cur = conn.execute(f"SELECT * FROM receipts WHERE {where} ORDER BY ref", params)
        return _rows(cur.fetchall())
    finally:
        conn.close()


def next_receipt_ref(db_path: Path | str, owner_id: int | None = None) -> int:
    conn = _connect(db_path)
    try:
        where, params = _scope(None, owner_id)
        row = conn.execute(
            f"SELECT COALESCE(MAX(ref), 0) AS m FROM receipts WHERE {where}",
            params,
        ).fetchone()
        return int(row["m"]) + 1
    finally:
        conn.close()


def add_receipt(db_path: Path | str, data: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        owner = _resolve_owner_id(conn, data.get("owner_id"))
        conn.execute(
            """
            INSERT INTO receipts (
                ref, date, description, category, project_code, project_name,
                amount, image_bytes, image_mime, month_slug, owner_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(data["ref"]),
                str(data["date"]),
                str(data["description"]),
                str(data["category"]),
                str(data.get("project_code") or ""),
                str(data.get("project_name") or ""),
                float(data["amount"]),
                data.get("image_bytes"),
                data.get("image_mime") or "image/jpeg",
                data.get("month_slug"),
                owner,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_receipt(db_path: Path | str, ref: int, owner_id: int | None = None) -> bool:
    conn = _connect(db_path)
    try:
        where, params = _scope(None, owner_id)
        cur = conn.execute(
            f"DELETE FROM receipts WHERE ref = ? AND {where}",
            [ref, *params],
        )
        if cur.rowcount <= 0:
            conn.commit()
            return False
        _renumber_current_receipts(conn, owner_id)
        conn.commit()
        return True
    finally:
        conn.close()


def _renumber_current_receipts(conn, owner_id: int | None = None) -> None:
    where, params = _scope(None, owner_id)
    rows = conn.execute(
        f"SELECT id FROM receipts WHERE {where} ORDER BY ref, id",
        params,
    ).fetchall()
    for i, row in enumerate(rows, start=1):
        conn.execute("UPDATE receipts SET ref = ? WHERE id = ?", (-i, row["id"]))
    for i, row in enumerate(rows, start=1):
        conn.execute("UPDATE receipts SET ref = ? WHERE id = ?", (i, row["id"]))


def list_credit_card(
    db_path: Path | str, month_slug: str | None, owner_id: int | None = None
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        where, params = _scope(month_slug, owner_id)
        cur = conn.execute(
            f"SELECT * FROM credit_card WHERE {where} ORDER BY COALESCE(ref, id), id",
            params,
        )
        return _rows(cur.fetchall())
    finally:
        conn.close()


def next_credit_card_ref(db_path: Path | str, owner_id: int | None = None) -> int:
    conn = _connect(db_path)
    try:
        where, params = _scope(None, owner_id)
        row = conn.execute(
            f"SELECT COALESCE(MAX(ref), 0) AS m FROM credit_card WHERE {where}",
            params,
        ).fetchone()
        return int(row["m"] or 0) + 1
    finally:
        conn.close()


def add_credit_card(db_path: Path | str, data: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        owner = _resolve_owner_id(conn, data.get("owner_id"))
        ref = data.get("ref")
        if ref is None:
            where, params = _scope(None, owner)
            row = conn.execute(
                f"SELECT COALESCE(MAX(ref), 0) AS m FROM credit_card WHERE {where}",
                params,
            ).fetchone()
            ref = int(row["m"] or 0) + 1
        conn.execute(
            """
            INSERT INTO credit_card (
                ref, date, description, category, project_code, project_name,
                amount, image_bytes, image_mime, month_slug, owner_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(ref),
                str(data["date"]),
                str(data["description"]),
                str(data["category"]),
                str(data.get("project_code") or ""),
                str(data.get("project_name") or ""),
                float(data["amount"]),
                data.get("image_bytes"),
                data.get("image_mime") or "image/jpeg",
                data.get("month_slug"),
                owner,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_credit_card(db_path: Path | str, row_id: int, owner_id: int | None = None) -> bool:
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        params: list = [row_id]
        if owner_id is not None:
            params.append(owner_id)
        cur = conn.execute(
            f"DELETE FROM credit_card WHERE id = ? AND month_slug IS NULL{extra}",
            params,
        )
        if cur.rowcount <= 0:
            conn.commit()
            return False
        where, wparams = _scope(None, owner_id)
        rows = conn.execute(
            f"SELECT id FROM credit_card WHERE {where} ORDER BY COALESCE(ref, id), id",
            wparams,
        ).fetchall()
        for i, row in enumerate(rows, start=1):
            conn.execute("UPDATE credit_card SET ref = ? WHERE id = ?", (-i, row["id"]))
        for i, row in enumerate(rows, start=1):
            conn.execute("UPDATE credit_card SET ref = ? WHERE id = ?", (i, row["id"]))
        conn.commit()
        return True
    finally:
        conn.close()


def list_transport(
    db_path: Path | str, month_slug: str | None, owner_id: int | None = None
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        where, params = _scope(month_slug, owner_id)
        cur = conn.execute(
            f"SELECT * FROM transport WHERE {where} ORDER BY date, id",
            params,
        )
        return _rows(cur.fetchall())
    finally:
        conn.close()


def add_transport(db_path: Path | str, data: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        owner = _resolve_owner_id(conn, data.get("owner_id"))
        conn.execute(
            """
            INSERT INTO transport (
                date, from_location, destination, return_included,
                project_code, project_name, month_slug, owner_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(data["date"]),
                str(data["from_location"]),
                str(data["destination"]),
                1 if data.get("return_included") else 0,
                str(data.get("project_code") or ""),
                str(data.get("project_name") or ""),
                data.get("month_slug"),
                owner,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_transport(db_path: Path | str, row_id: int, owner_id: int | None = None) -> bool:
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        params: list = [row_id]
        if owner_id is not None:
            params.append(owner_id)
        cur = conn.execute(
            f"DELETE FROM transport WHERE id = ? AND month_slug IS NULL{extra}",
            params,
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def list_archived_months(db_path: Path | str, owner_id: int | None = None) -> list[str]:
    conn = _connect(db_path)
    try:
        months: set[str] = set()
        extra = " AND owner_id = ?" if owner_id is not None else ""
        params = (owner_id,) if owner_id is not None else ()
        for table in ("receipts", "credit_card", "transport"):
            cur = conn.execute(
                f"SELECT DISTINCT month_slug FROM {table} WHERE month_slug IS NOT NULL AND month_slug != ''{extra}",
                params,
            )
            for row in cur.fetchall():
                months.add(row["month_slug"])
        return sorted(months, reverse=True)
    finally:
        conn.close()


def archive_current_month(db_path: Path | str, month_slug: str, owner_id: int | None = None) -> None:
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        params = (month_slug, owner_id) if owner_id is not None else (month_slug,)
        existing = conn.execute(
            f"SELECT 1 FROM receipts WHERE month_slug = ?{extra} LIMIT 1",
            params,
        ).fetchone()
        if existing:
            raise RuntimeError(f"Archive already exists for {month_slug}")
        counts = 0
        open_params = (owner_id,) if owner_id is not None else ()
        for table in ("receipts", "credit_card", "transport"):
            counts += conn.execute(
                f"SELECT COUNT(*) AS c FROM {table} WHERE month_slug IS NULL{extra}",
                open_params,
            ).fetchone()["c"]
        if counts == 0:
            raise RuntimeError("No data to archive. Current month is empty.")
        upd_params = (month_slug, owner_id) if owner_id is not None else (month_slug,)
        for table in ("receipts", "credit_card", "transport"):
            conn.execute(
                f"UPDATE {table} SET month_slug = ? WHERE month_slug IS NULL{extra}",
                upd_params,
            )
        conn.commit()
    finally:
        conn.close()


def _count(conn, sql: str, params: tuple = ()) -> int:
    return int(conn.execute(sql, params).fetchone()["c"])


def working_month_stats(db_path: Path | str, owner_id: int | None = None) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        params = (owner_id,) if owner_id is not None else ()
        receipts_total = conn.execute(
            f"SELECT COALESCE(SUM(amount), 0) AS t FROM receipts WHERE month_slug IS NULL{extra}",
            params,
        ).fetchone()["t"]
        cc_total = conn.execute(
            f"SELECT COALESCE(SUM(amount), 0) AS t FROM credit_card WHERE month_slug IS NULL{extra}",
            params,
        ).fetchone()["t"]
        drafts_sql = "SELECT COUNT(*) AS c FROM drafts"
        drafts_params: tuple = ()
        if owner_id is not None:
            user = conn.execute(
                "SELECT telegram_user_id FROM users WHERE id = ?", (owner_id,)
            ).fetchone()
            tg = user["telegram_user_id"] if user else None
            if tg:
                drafts_sql = "SELECT COUNT(*) AS c FROM drafts WHERE chat_id = ?"
                drafts_params = (str(tg),)
            else:
                drafts_sql = "SELECT 0 AS c"
        return {
            "receipts": _count(conn, f"SELECT COUNT(*) AS c FROM receipts WHERE month_slug IS NULL{extra}", params),
            "receipts_total": float(receipts_total or 0),
            "receipt_photos": _count(
                conn,
                f"SELECT COUNT(*) AS c FROM receipts WHERE month_slug IS NULL AND image_bytes IS NOT NULL{extra}",
                params,
            ),
            "credit_card": _count(
                conn, f"SELECT COUNT(*) AS c FROM credit_card WHERE month_slug IS NULL{extra}", params
            ),
            "credit_card_total": float(cc_total or 0),
            "credit_card_photos": _count(
                conn,
                f"SELECT COUNT(*) AS c FROM credit_card WHERE month_slug IS NULL AND image_bytes IS NOT NULL{extra}",
                params,
            ),
            "transport": _count(
                conn, f"SELECT COUNT(*) AS c FROM transport WHERE month_slug IS NULL{extra}", params
            ),
            "drafts": _count(conn, drafts_sql, drafts_params),
            "archived_months": len(list_archived_months(db_path, owner_id)),
        }
    finally:
        conn.close()


def clear_current_month(db_path: Path | str, owner_id: int | None = None) -> dict[str, int]:
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        params = (owner_id,) if owner_id is not None else ()
        counts = {
            "receipts": _count(conn, f"SELECT COUNT(*) AS c FROM receipts WHERE month_slug IS NULL{extra}", params),
            "credit_card": _count(
                conn, f"SELECT COUNT(*) AS c FROM credit_card WHERE month_slug IS NULL{extra}", params
            ),
            "transport": _count(
                conn, f"SELECT COUNT(*) AS c FROM transport WHERE month_slug IS NULL{extra}", params
            ),
        }
        for table in ("receipts", "credit_card", "transport"):
            conn.execute(f"DELETE FROM {table} WHERE month_slug IS NULL{extra}", params)
        conn.commit()
        return counts
    finally:
        conn.close()


def delete_archived_month(
    db_path: Path | str, month_slug: str, owner_id: int | None = None
) -> dict[str, int]:
    slug = (month_slug or "").strip()
    if not slug:
        raise RuntimeError("No archived month selected.")
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        counts = {}
        for table in ("receipts", "credit_card", "transport"):
            params = (slug, owner_id) if owner_id is not None else (slug,)
            counts[table] = _count(
                conn,
                f"SELECT COUNT(*) AS c FROM {table} WHERE month_slug = ?{extra}",
                params,
            )
            conn.execute(f"DELETE FROM {table} WHERE month_slug = ?{extra}", params)
        if sum(counts.values()) == 0:
            raise RuntimeError(f"No archived data for {slug.replace('_', ' ')}")
        conn.commit()
        return counts
    finally:
        conn.close()


def clear_all_drafts(db_path: Path | str) -> int:
    conn = _connect(db_path)
    try:
        count = _count(conn, "SELECT COUNT(*) AS c FROM drafts")
        conn.execute("DELETE FROM drafts")
        conn.commit()
        return count
    finally:
        conn.close()


def archived_month_stats(db_path: Path | str, owner_id: int | None = None) -> list[dict[str, Any]]:
    months = list_archived_months(db_path, owner_id)
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        out = []
        for slug in months:
            params = (slug, owner_id) if owner_id is not None else (slug,)
            receipts_total = conn.execute(
                f"SELECT COALESCE(SUM(amount), 0) AS t FROM receipts WHERE month_slug = ?{extra}",
                params,
            ).fetchone()["t"]
            out.append(
                {
                    "slug": slug,
                    "label": slug.replace("_", " "),
                    "receipts": _count(
                        conn, f"SELECT COUNT(*) AS c FROM receipts WHERE month_slug = ?{extra}", params
                    ),
                    "receipts_total": float(receipts_total or 0),
                    "credit_card": _count(
                        conn, f"SELECT COUNT(*) AS c FROM credit_card WHERE month_slug = ?{extra}", params
                    ),
                    "transport": _count(
                        conn, f"SELECT COUNT(*) AS c FROM transport WHERE month_slug = ?{extra}", params
                    ),
                }
            )
        return out
    finally:
        conn.close()


def project_codes_for_current(db_path: Path | str, owner_id: int | None = None) -> list[str]:
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        params = (owner_id,) if owner_id is not None else ()
        codes: set[str] = set()
        for table, col in (("receipts", "project_code"), ("credit_card", "project_code"), ("transport", "project_code")):
            cur = conn.execute(
                f"SELECT DISTINCT {col} FROM {table} WHERE month_slug IS NULL AND {col} IS NOT NULL AND {col} != ''{extra}",
                params,
            )
            for row in cur.fetchall():
                codes.add(str(row[0]).strip())
        return sorted(codes)
    finally:
        conn.close()


def project_names(db_path: Path | str, owner_id: int | None = None) -> list[str]:
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        params = (owner_id,) if owner_id is not None else ()
        names: set[str] = set()
        for table in ("receipts", "credit_card", "transport"):
            cur = conn.execute(
                f"SELECT DISTINCT project_name FROM {table} WHERE project_name IS NOT NULL AND project_name != ''{extra}",
                params,
            )
            for row in cur.fetchall():
                names.add(str(row[0]).strip())
        return sorted(names, key=str.lower)
    finally:
        conn.close()


def get_receipt(
    db_path: Path | str, ref: int, month_slug: str | None, owner_id: int | None = None
) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        where, params = _scope(month_slug, owner_id)
        row = conn.execute(
            f"SELECT * FROM receipts WHERE ref = ? AND {where}",
            [ref, *params],
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_receipt(
    db_path: Path | str,
    ref: int,
    data: dict[str, Any],
    replace_image: bool = False,
    owner_id: int | None = None,
) -> bool:
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        owner_params = (owner_id,) if owner_id is not None else ()
        if replace_image:
            cur = conn.execute(
                f"""
                UPDATE receipts SET
                    date = ?, description = ?, category = ?, project_code = ?,
                    project_name = ?, amount = ?, image_bytes = ?, image_mime = ?
                WHERE ref = ? AND month_slug IS NULL{extra}
                """,
                (
                    str(data["date"]),
                    str(data["description"]),
                    str(data["category"]),
                    str(data.get("project_code") or ""),
                    str(data.get("project_name") or ""),
                    float(data["amount"]),
                    data.get("image_bytes"),
                    data.get("image_mime") or "image/jpeg",
                    ref,
                    *owner_params,
                ),
            )
        else:
            cur = conn.execute(
                f"""
                UPDATE receipts SET
                    date = ?, description = ?, category = ?, project_code = ?,
                    project_name = ?, amount = ?
                WHERE ref = ? AND month_slug IS NULL{extra}
                """,
                (
                    str(data["date"]),
                    str(data["description"]),
                    str(data["category"]),
                    str(data.get("project_code") or ""),
                    str(data.get("project_name") or ""),
                    float(data["amount"]),
                    ref,
                    *owner_params,
                ),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_credit_card(
    db_path: Path | str, row_id: int, month_slug: str | None, owner_id: int | None = None
) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        where, params = _scope(month_slug, owner_id)
        row = conn.execute(
            f"SELECT * FROM credit_card WHERE id = ? AND {where}",
            [row_id, *params],
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_credit_card(
    db_path: Path | str,
    row_id: int,
    data: dict[str, Any],
    replace_image: bool = False,
    owner_id: int | None = None,
) -> bool:
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        owner_params = (owner_id,) if owner_id is not None else ()
        if replace_image:
            cur = conn.execute(
                f"""
                UPDATE credit_card SET
                    date = ?, description = ?, category = ?, project_code = ?,
                    project_name = ?, amount = ?, image_bytes = ?, image_mime = ?
                WHERE id = ? AND month_slug IS NULL{extra}
                """,
                (
                    str(data["date"]),
                    str(data["description"]),
                    str(data["category"]),
                    str(data.get("project_code") or ""),
                    str(data.get("project_name") or ""),
                    float(data["amount"]),
                    data.get("image_bytes"),
                    data.get("image_mime") or "image/jpeg",
                    row_id,
                    *owner_params,
                ),
            )
        else:
            cur = conn.execute(
                f"""
                UPDATE credit_card SET
                    date = ?, description = ?, category = ?, project_code = ?,
                    project_name = ?, amount = ?
                WHERE id = ? AND month_slug IS NULL{extra}
                """,
                (
                    str(data["date"]),
                    str(data["description"]),
                    str(data["category"]),
                    str(data.get("project_code") or ""),
                    str(data.get("project_name") or ""),
                    float(data["amount"]),
                    row_id,
                    *owner_params,
                ),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_transport(
    db_path: Path | str, row_id: int, month_slug: str | None, owner_id: int | None = None
) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        where, params = _scope(month_slug, owner_id)
        row = conn.execute(
            f"SELECT * FROM transport WHERE id = ? AND {where}",
            [row_id, *params],
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_transport(
    db_path: Path | str, row_id: int, data: dict[str, Any], owner_id: int | None = None
) -> bool:
    conn = _connect(db_path)
    try:
        extra = " AND owner_id = ?" if owner_id is not None else ""
        owner_params = (owner_id,) if owner_id is not None else ()
        cur = conn.execute(
            f"""
            UPDATE transport SET
                date = ?, from_location = ?, destination = ?, return_included = ?,
                project_code = ?, project_name = ?
            WHERE id = ? AND month_slug IS NULL{extra}
            """,
            (
                str(data["date"]),
                str(data["from_location"]),
                str(data["destination"]),
                1 if data.get("return_included") else 0,
                str(data.get("project_code") or ""),
                str(data.get("project_name") or ""),
                row_id,
                *owner_params,
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_draft(db_path: Path | str, chat_id: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM drafts WHERE chat_id = ?", (str(chat_id),)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def upsert_draft(
    db_path: Path | str,
    chat_id: str,
    payload: str,
    image_bytes: bytes | None = None,
    image_mime: str | None = None,
) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO drafts (chat_id, payload, image_bytes, image_mime, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chat_id) DO UPDATE SET
                payload = excluded.payload,
                image_bytes = COALESCE(excluded.image_bytes, drafts.image_bytes),
                image_mime = COALESCE(excluded.image_mime, drafts.image_mime),
                updated_at = CURRENT_TIMESTAMP
            """,
            (str(chat_id), payload, image_bytes, image_mime),
        )
        conn.commit()
    finally:
        conn.close()


def delete_draft(db_path: Path | str, chat_id: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM drafts WHERE chat_id = ?", (str(chat_id),))
        conn.commit()
    finally:
        conn.close()


def import_sqlite_file(src_path: Path | str, dest_path: Path | str | None = None) -> None:
    """Copy all expense tables from a local SQLite file into the active database (Turso or local)."""
    src = sqlite3.connect(src_path)
    src.row_factory = sqlite3.Row
    conn = _connect(dest_path or src_path)
    try:
        conn.executescript(_SCHEMA)
        src_tables = {
            r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        for table in ("users", "receipts", "credit_card", "transport", "drafts"):
            if table not in src_tables:
                continue
            conn.execute(f"DELETE FROM {table}")
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                continue
            cols = list(rows[0].keys())
            placeholders = ",".join("?" * len(cols))
            col_list = ",".join(cols)
            sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
            for row in rows:
                conn.execute(sql, [row[c] for c in cols])
        conn.commit()
    finally:
        src.close()
        conn.close()


def export_sqlite_file(dest_path: Path | str) -> Path:
    """Write the active database (Turso or local) to a SQLite file."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not turso_enabled():
        src = Path(settings.database_path)
        if not src.exists():
            raise RuntimeError("No database yet.")
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
        return dest
    src_conn = _connect(settings.database_path, persist=False)
    out = sqlite3.connect(dest)
    try:
        out.executescript(_SCHEMA)
        for table in ("receipts", "credit_card", "transport", "drafts"):
            out.execute(f"DELETE FROM {table}")
            rows = src_conn.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                continue
            cols = list(rows[0].keys())
            placeholders = ",".join("?" * len(cols))
            col_list = ",".join(cols)
            sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
            for row in rows:
                out.execute(sql, [row[c] for c in cols])
        out.commit()
    finally:
        src_conn.close()
        out.close()
    return dest

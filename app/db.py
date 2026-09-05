from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

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


def turso_enabled() -> bool:
    return bool(settings.turso_database_url and settings.turso_auth_token)


def _connect(db_path: Path | str):
    if turso_enabled():
        import libsql

        conn = libsql.connect(settings.turso_database_url, auth_token=settings.turso_auth_token)
        return _CompatConn(conn)
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


_SCHEMA = """
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
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        _migrate_credit_card(conn)
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


def _rows(cursor_result) -> list[dict[str, Any]]:
    return [dict(r) for r in cursor_result]


def list_receipts(db_path: Path | str, month_slug: str | None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        if month_slug:
            cur = conn.execute(
                "SELECT * FROM receipts WHERE month_slug = ? ORDER BY ref",
                (month_slug,),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM receipts WHERE month_slug IS NULL ORDER BY ref"
            )
        return _rows(cur.fetchall())
    finally:
        conn.close()


def next_receipt_ref(db_path: Path | str) -> int:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(ref), 0) AS m FROM receipts WHERE month_slug IS NULL"
        ).fetchone()
        return int(row["m"]) + 1
    finally:
        conn.close()


def add_receipt(db_path: Path | str, data: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO receipts (
                ref, date, description, category, project_code, project_name,
                amount, image_bytes, image_mime, month_slug
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_receipt(db_path: Path | str, ref: int) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "DELETE FROM receipts WHERE ref = ? AND month_slug IS NULL",
            (ref,),
        )
        if cur.rowcount <= 0:
            conn.commit()
            return False
        _renumber_current_receipts(conn)
        conn.commit()
        return True
    finally:
        conn.close()


def _renumber_current_receipts(conn) -> None:
    rows = conn.execute(
        "SELECT id FROM receipts WHERE month_slug IS NULL ORDER BY ref, id"
    ).fetchall()
    for i, row in enumerate(rows, start=1):
        conn.execute("UPDATE receipts SET ref = ? WHERE id = ?", (-i, row["id"]))
    for i, row in enumerate(rows, start=1):
        conn.execute("UPDATE receipts SET ref = ? WHERE id = ?", (i, row["id"]))


def list_credit_card(db_path: Path | str, month_slug: str | None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        if month_slug:
            cur = conn.execute(
                "SELECT * FROM credit_card WHERE month_slug = ? ORDER BY COALESCE(ref, id), id",
                (month_slug,),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM credit_card WHERE month_slug IS NULL ORDER BY COALESCE(ref, id), id"
            )
        return _rows(cur.fetchall())
    finally:
        conn.close()


def next_credit_card_ref(db_path: Path | str) -> int:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(ref), 0) AS m FROM credit_card WHERE month_slug IS NULL"
        ).fetchone()
        return int(row["m"] or 0) + 1
    finally:
        conn.close()


def add_credit_card(db_path: Path | str, data: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        ref = data.get("ref")
        if ref is None:
            row = conn.execute(
                "SELECT COALESCE(MAX(ref), 0) AS m FROM credit_card WHERE month_slug IS NULL"
            ).fetchone()
            ref = int(row["m"] or 0) + 1
        conn.execute(
            """
            INSERT INTO credit_card (
                ref, date, description, category, project_code, project_name,
                amount, image_bytes, image_mime, month_slug
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_credit_card(db_path: Path | str, row_id: int) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "DELETE FROM credit_card WHERE id = ? AND month_slug IS NULL",
            (row_id,),
        )
        if cur.rowcount <= 0:
            conn.commit()
            return False
        rows = conn.execute(
            "SELECT id FROM credit_card WHERE month_slug IS NULL ORDER BY COALESCE(ref, id), id"
        ).fetchall()
        for i, row in enumerate(rows, start=1):
            conn.execute("UPDATE credit_card SET ref = ? WHERE id = ?", (-i, row["id"]))
        for i, row in enumerate(rows, start=1):
            conn.execute("UPDATE credit_card SET ref = ? WHERE id = ?", (i, row["id"]))
        conn.commit()
        return True
    finally:
        conn.close()


def list_transport(db_path: Path | str, month_slug: str | None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        if month_slug:
            cur = conn.execute(
                "SELECT * FROM transport WHERE month_slug = ? ORDER BY date, id",
                (month_slug,),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM transport WHERE month_slug IS NULL ORDER BY date, id"
            )
        return _rows(cur.fetchall())
    finally:
        conn.close()


def add_transport(db_path: Path | str, data: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO transport (
                date, from_location, destination, return_included,
                project_code, project_name, month_slug
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(data["date"]),
                str(data["from_location"]),
                str(data["destination"]),
                1 if data.get("return_included") else 0,
                str(data.get("project_code") or ""),
                str(data.get("project_name") or ""),
                data.get("month_slug"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_transport(db_path: Path | str, row_id: int) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "DELETE FROM transport WHERE id = ? AND month_slug IS NULL",
            (row_id,),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def list_archived_months(db_path: Path | str) -> list[str]:
    conn = _connect(db_path)
    try:
        months: set[str] = set()
        for table in ("receipts", "credit_card", "transport"):
            cur = conn.execute(
                f"SELECT DISTINCT month_slug FROM {table} WHERE month_slug IS NOT NULL AND month_slug != ''"
            )
            for row in cur.fetchall():
                months.add(row["month_slug"])
        return sorted(months, reverse=True)
    finally:
        conn.close()


def archive_current_month(db_path: Path | str, month_slug: str) -> None:
    conn = _connect(db_path)
    try:
        existing = conn.execute(
            "SELECT 1 FROM receipts WHERE month_slug = ? LIMIT 1", (month_slug,)
        ).fetchone()
        if existing:
            raise RuntimeError(f"Archive already exists for {month_slug}")
        counts = 0
        for table in ("receipts", "credit_card", "transport"):
            counts += conn.execute(
                f"SELECT COUNT(*) AS c FROM {table} WHERE month_slug IS NULL"
            ).fetchone()["c"]
        if counts == 0:
            raise RuntimeError("No data to archive. Current month is empty.")
        for table in ("receipts", "credit_card", "transport"):
            conn.execute(
                f"UPDATE {table} SET month_slug = ? WHERE month_slug IS NULL",
                (month_slug,),
            )
        conn.commit()
    finally:
        conn.close()


def project_codes_for_current(db_path: Path | str) -> list[str]:
    conn = _connect(db_path)
    try:
        codes: set[str] = set()
        for table, col in (("receipts", "project_code"), ("credit_card", "project_code"), ("transport", "project_code")):
            cur = conn.execute(
                f"SELECT DISTINCT {col} FROM {table} WHERE month_slug IS NULL AND {col} IS NOT NULL AND {col} != ''"
            )
            for row in cur.fetchall():
                codes.add(str(row[0]).strip())
        return sorted(codes)
    finally:
        conn.close()


def project_names(db_path: Path | str) -> list[str]:
    conn = _connect(db_path)
    try:
        names: set[str] = set()
        for table in ("receipts", "credit_card", "transport"):
            cur = conn.execute(
                f"SELECT DISTINCT project_name FROM {table} WHERE project_name IS NOT NULL AND project_name != ''"
            )
            for row in cur.fetchall():
                names.add(str(row[0]).strip())
        return sorted(names, key=str.lower)
    finally:
        conn.close()


def get_receipt(db_path: Path | str, ref: int, month_slug: str | None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        if month_slug:
            row = conn.execute(
                "SELECT * FROM receipts WHERE ref = ? AND month_slug = ?",
                (ref, month_slug),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM receipts WHERE ref = ? AND month_slug IS NULL",
                (ref,),
            ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_receipt(db_path: Path | str, ref: int, data: dict[str, Any], replace_image: bool = False) -> bool:
    conn = _connect(db_path)
    try:
        if replace_image:
            cur = conn.execute(
                """
                UPDATE receipts SET
                    date = ?, description = ?, category = ?, project_code = ?,
                    project_name = ?, amount = ?, image_bytes = ?, image_mime = ?
                WHERE ref = ? AND month_slug IS NULL
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
                ),
            )
        else:
            cur = conn.execute(
                """
                UPDATE receipts SET
                    date = ?, description = ?, category = ?, project_code = ?,
                    project_name = ?, amount = ?
                WHERE ref = ? AND month_slug IS NULL
                """,
                (
                    str(data["date"]),
                    str(data["description"]),
                    str(data["category"]),
                    str(data.get("project_code") or ""),
                    str(data.get("project_name") or ""),
                    float(data["amount"]),
                    ref,
                ),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_credit_card(db_path: Path | str, row_id: int, month_slug: str | None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        if month_slug:
            row = conn.execute(
                "SELECT * FROM credit_card WHERE id = ? AND month_slug = ?",
                (row_id, month_slug),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM credit_card WHERE id = ? AND month_slug IS NULL",
                (row_id,),
            ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_credit_card(db_path: Path | str, row_id: int, data: dict[str, Any], replace_image: bool = False) -> bool:
    conn = _connect(db_path)
    try:
        if replace_image:
            cur = conn.execute(
                """
                UPDATE credit_card SET
                    date = ?, description = ?, category = ?, project_code = ?,
                    project_name = ?, amount = ?, image_bytes = ?, image_mime = ?
                WHERE id = ? AND month_slug IS NULL
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
                ),
            )
        else:
            cur = conn.execute(
                """
                UPDATE credit_card SET
                    date = ?, description = ?, category = ?, project_code = ?,
                    project_name = ?, amount = ?
                WHERE id = ? AND month_slug IS NULL
                """,
                (
                    str(data["date"]),
                    str(data["description"]),
                    str(data["category"]),
                    str(data.get("project_code") or ""),
                    str(data.get("project_name") or ""),
                    float(data["amount"]),
                    row_id,
                ),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_transport(db_path: Path | str, row_id: int, month_slug: str | None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        if month_slug:
            row = conn.execute(
                "SELECT * FROM transport WHERE id = ? AND month_slug = ?",
                (row_id, month_slug),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM transport WHERE id = ? AND month_slug IS NULL",
                (row_id,),
            ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_transport(db_path: Path | str, row_id: int, data: dict[str, Any]) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            UPDATE transport SET
                date = ?, from_location = ?, destination = ?, return_included = ?,
                project_code = ?, project_name = ?
            WHERE id = ? AND month_slug IS NULL
            """,
            (
                str(data["date"]),
                str(data["from_location"]),
                str(data["destination"]),
                1 if data.get("return_included") else 0,
                str(data.get("project_code") or ""),
                str(data.get("project_name") or ""),
                row_id,
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
        for table in ("receipts", "credit_card", "transport", "drafts"):
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

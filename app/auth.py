from __future__ import annotations

import hashlib
import hmac
import re
import secrets


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), 120_000)
    return f"pbkdf2${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        kind, salt, digest = stored.split("$", 2)
    except ValueError:
        return False
    if kind != "pbkdf2":
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), 120_000)
    return hmac.compare_digest(check.hex(), digest)


def username_from_name(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    return (words[0] if words else "user")[:24]


def unique_username(existing: set[str], name: str) -> str:
    base = username_from_name(name) or "user"
    if base not in existing:
        return base
    n = 2
    while f"{base}{n}" in existing:
        n += 1
    return f"{base}{n}"


def new_invite_code() -> str:
    return secrets.token_hex(3).upper()

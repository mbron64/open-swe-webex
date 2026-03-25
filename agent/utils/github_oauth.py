"""Custom GitHub OAuth flow for per-user access tokens.

Handles the OAuth web application flow for GitHub Apps, storing encrypted
user tokens so each Webex user operates with their own GitHub permissions.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path
from threading import Lock
from typing import Any

import httpx

from ..encryption import decrypt_token, encrypt_token

logger = logging.getLogger(__name__)

GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
GITHUB_OAUTH_CALLBACK_URL = os.environ.get("GITHUB_OAUTH_CALLBACK_URL", "")

_TOKEN_FILE = Path(
    os.environ.get(
        "GITHUB_USER_TOKENS_PATH",
        str(Path(__file__).resolve().parent.parent.parent / ".data" / "github_user_tokens.json"),
    )
)
_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

_file_lock = Lock()


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    code_verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def get_github_oauth_url(email: str, room_id: str, parent_id: str | None = None) -> str | None:
    """Build the GitHub OAuth authorization URL.

    Encodes the user's Webex identity and PKCE verifier in the encrypted state
    parameter so the callback can map the token back to the right user.

    Returns None if GitHub OAuth is not configured.
    """
    if not GITHUB_CLIENT_ID:
        logger.warning("GITHUB_CLIENT_ID not set, cannot generate OAuth URL")
        return None

    code_verifier, code_challenge = _generate_pkce()

    state_payload = json.dumps(
        {
            "email": email,
            "room_id": room_id,
            "parent_id": parent_id or "",
            "code_verifier": code_verifier,
            "ts": int(time.time()),
        }
    )
    encrypted_state = encrypt_token(state_payload)

    params = {
        "client_id": GITHUB_CLIENT_ID,
        "state": encrypted_state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "allow_signup": "false",
    }
    if GITHUB_OAUTH_CALLBACK_URL:
        params["redirect_uri"] = GITHUB_OAUTH_CALLBACK_URL

    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"https://github.com/login/oauth/authorize?{query}"


def decrypt_oauth_state(encrypted_state: str) -> dict[str, Any] | None:
    """Decrypt and parse the OAuth state parameter.

    Returns None if decryption fails or the state is too old (> 15 minutes).
    """
    try:
        decrypted = decrypt_token(encrypted_state)
        if not decrypted:
            return None
        data = json.loads(decrypted)
        if time.time() - data.get("ts", 0) > 900:
            logger.warning("OAuth state expired (older than 15 minutes)")
            return None
        return data
    except (json.JSONDecodeError, Exception):
        logger.warning("Failed to decrypt/parse OAuth state", exc_info=True)
        return None


async def exchange_code_for_tokens(code: str, code_verifier: str) -> dict[str, Any] | None:
    """Exchange an OAuth authorization code for access and refresh tokens."""
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        logger.error("GITHUB_CLIENT_ID or GITHUB_CLIENT_SECRET not configured")
        return None

    payload = {
        "client_id": GITHUB_CLIENT_ID,
        "client_secret": GITHUB_CLIENT_SECRET,
        "code": code,
        "code_verifier": code_verifier,
    }
    if GITHUB_OAUTH_CALLBACK_URL:
        payload["redirect_uri"] = GITHUB_OAUTH_CALLBACK_URL

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://github.com/login/oauth/access_token",
                data=payload,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

        if "error" in data:
            logger.error(
                "GitHub OAuth token exchange failed: %s - %s",
                data.get("error"),
                data.get("error_description"),
            )
            return None

        return data
    except Exception:
        logger.exception("GitHub OAuth token exchange request failed")
        return None


async def refresh_access_token(refresh_token: str) -> dict[str, Any] | None:
    """Use a refresh token to obtain a new access token."""
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

        if "error" in data:
            logger.error(
                "GitHub OAuth token refresh failed: %s - %s",
                data.get("error"),
                data.get("error_description"),
            )
            return None

        return data
    except Exception:
        logger.exception("GitHub OAuth token refresh request failed")
        return None


def _read_token_store() -> dict[str, Any]:
    """Read the encrypted token store from disk."""
    if not _TOKEN_FILE.exists():
        return {}
    try:
        return json.loads(_TOKEN_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read token store at %s", _TOKEN_FILE, exc_info=True)
        return {}


def _write_token_store(store: dict[str, Any]) -> None:
    """Atomically write the token store to disk."""
    tmp = _TOKEN_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=2))
    tmp.rename(_TOKEN_FILE)


def _store_user_tokens_sync(
    email: str,
    access_token: str,
    refresh_token: str,
    expires_in: int | None,
) -> None:
    """Encrypt and persist a user's GitHub tokens (sync, runs in thread)."""
    expires_at = time.time() + expires_in if expires_in else 0.0

    entry = {
        "access_token": encrypt_token(access_token),
        "refresh_token": encrypt_token(refresh_token) if refresh_token else "",
        "expires_at": expires_at,
    }

    with _file_lock:
        store = _read_token_store()
        store[email.lower()] = entry
        _write_token_store(store)

    logger.info("Stored GitHub OAuth tokens for %s (expires_at=%.0f)", email, expires_at)


async def store_user_tokens(
    email: str,
    access_token: str,
    refresh_token: str,
    expires_in: int | None,
) -> None:
    """Encrypt and persist a user's GitHub tokens."""
    import asyncio

    await asyncio.to_thread(_store_user_tokens_sync, email, access_token, refresh_token, expires_in)


def _get_user_token_sync(email: str) -> dict[str, Any] | None:
    """Read a user's token entry from the store (sync, runs in thread)."""
    with _file_lock:
        store = _read_token_store()
    return store.get(email.lower())


async def get_user_token(email: str) -> str | None:
    """Retrieve a valid GitHub access token for a user.

    Automatically refreshes expired tokens. Returns None if no token is
    stored or if the refresh token has also expired.
    """
    import asyncio

    entry = await asyncio.to_thread(_get_user_token_sync, email)
    if not entry:
        return None

    access_token = decrypt_token(entry.get("access_token", ""))
    if not access_token:
        return None

    expires_at = entry.get("expires_at", 0.0)
    if expires_at and time.time() < expires_at - 60:
        return access_token

    stored_refresh = decrypt_token(entry.get("refresh_token", ""))
    if not stored_refresh:
        logger.info("Access token expired and no refresh token for %s", email)
        await asyncio.to_thread(_remove_user_tokens, email)
        return None

    logger.info("Access token expired for %s, refreshing", email)
    refreshed = await refresh_access_token(stored_refresh)
    if not refreshed or "access_token" not in refreshed:
        logger.warning("Token refresh failed for %s, clearing stored tokens", email)
        await asyncio.to_thread(_remove_user_tokens, email)
        return None

    await store_user_tokens(
        email,
        refreshed["access_token"],
        refreshed.get("refresh_token", stored_refresh),
        refreshed.get("expires_in"),
    )
    return refreshed["access_token"]


def _remove_user_tokens(email: str) -> None:
    """Remove a user's tokens from the store."""
    with _file_lock:
        store = _read_token_store()
        store.pop(email.lower(), None)
        _write_token_store(store)


def has_user_token(email: str) -> bool:
    """Check if a user has stored tokens (may still be expired)."""
    with _file_lock:
        store = _read_token_store()
    return email.lower() in store

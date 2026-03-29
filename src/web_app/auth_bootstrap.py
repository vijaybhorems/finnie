"""Bootstrap Streamlit auth secrets from environment variables.

On Cloud Run, secrets are injected as env vars (not via secrets.toml).
This module generates .streamlit/secrets.toml at startup from env vars.

For local development, create .streamlit/secrets.toml manually instead.
"""
from __future__ import annotations

import os
from pathlib import Path


def bootstrap_auth_secrets() -> None:
    """Generate .streamlit/secrets.toml from env vars if not present.

    Only runs when GOOGLE_CLIENT_ID is set (i.e., Cloud Run / production).
    """
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    if not client_id:
        return  # Use existing .streamlit/secrets.toml for local dev

    secrets_dir = Path("/app/.streamlit")
    secrets_file = secrets_dir / "secrets.toml"

    # Don't overwrite if already exists
    if secrets_file.exists():
        return

    secrets_dir.mkdir(parents=True, exist_ok=True)

    redirect_uri = os.environ.get("AUTH_REDIRECT_URI", "")
    cookie_secret = os.environ.get("AUTH_COOKIE_SECRET", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")

    content = f"""[auth]
redirect_uri = "{redirect_uri}"
cookie_secret = "{cookie_secret}"

[auth.google]
client_id = "{client_id}"
client_secret = "{client_secret}"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
"""

    secrets_file.write_text(content)

"""
Apache Superset configuration for the Enterprise GenAI Platform.
Enables SSO via Keycloak (OpenID Connect / OAuth2).
"""

import os

from flask_appbuilder.security.manager import AUTH_OAUTH

# ── Core ───────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SUPERSET_SECRET_KEY", "changeme-superset-secret")

SQLALCHEMY_DATABASE_URI = (
    f"postgresql+psycopg2://superset:superset123@"
    f"{os.getenv('DATABASE_HOST', 'postgres')}:"
    f"{os.getenv('DATABASE_PORT', '5432')}/"
    f"{os.getenv('DATABASE_DB', 'superset')}"
)

# ── Authentication — SSO via Keycloak ──────────────────────────────────────────
AUTH_TYPE = AUTH_OAUTH
AUTH_USER_REGISTRATION = True
AUTH_USER_REGISTRATION_ROLE = "Viewer"

_KC_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080")
_REALM  = "enterprise-genai"
_BASE   = f"{_KC_URL}/realms/{_REALM}/protocol/openid-connect"

OAUTH_PROVIDERS = [
    {
        "name": "keycloak",
        "token_key": "access_token",
        "icon": "fa-key",
        "remote_app": {
            "client_id":           os.getenv("SUPERSET_OAUTH_CLIENT_ID", "superset"),
            "client_secret":       os.getenv("SUPERSET_OAUTH_CLIENT_SECRET", "superset-secret"),
            "api_base_url":        _BASE,
            "access_token_url":    f"{_BASE}/token",
            "authorize_url":       f"{_BASE}/auth",
            "jwks_uri":            f"{_BASE}/certs",
            "userinfo_endpoint":   f"{_BASE}/userinfo",
            "client_kwargs":       {"scope": "openid email profile roles"},
            "token_endpoint_auth_method": "client_secret_post",
        },
    }
]

# Map Keycloak realm roles → Superset roles
AUTH_ROLES_MAPPING = {
    "admin":          ["Admin"],
    "data-engineer":  ["Alpha"],
    "data-scientist": ["Alpha"],
    "analyst":        ["Gamma"],
    "viewer":         ["Viewer"],
}
AUTH_ROLES_SYNC_AT_LOGIN = True

# ── Performance ────────────────────────────────────────────────────────────────
CACHE_CONFIG = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_DEFAULT_TIMEOUT": 300,
    "CACHE_KEY_PREFIX": "superset_",
    "CACHE_REDIS_URL": os.getenv("REDIS_URL", "redis://redis:6379/2"),
}

DATA_CACHE_CONFIG = {
    **CACHE_CONFIG,
    "CACHE_KEY_PREFIX": "superset_data_",
    "CACHE_DEFAULT_TIMEOUT": 600,
}

# ── Feature flags ──────────────────────────────────────────────────────────────
FEATURE_FLAGS = {
    "ENABLE_TEMPLATE_PROCESSING": True,
    "DASHBOARD_NATIVE_FILTERS": True,
    "DASHBOARD_CROSS_FILTERS": True,
    "ALERT_REPORTS": True,
    "EMBEDDED_SUPERSET": True,
}

# ── Security ───────────────────────────────────────────────────────────────────
SESSION_COOKIE_HTTPONLY    = True
SESSION_COOKIE_SECURE      = False   # set True in production (HTTPS)
SESSION_COOKIE_SAMESITE    = "Lax"
WTF_CSRF_ENABLED           = True
TALISMAN_ENABLED           = False   # enable in production with proper CSP

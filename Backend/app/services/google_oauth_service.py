
from authlib.integrations.starlette_client import OAuth

from app.core.config import get_settings

settings = get_settings()

oauth = OAuth()


def configure_google_oauth() -> None:
    """Register Google OAuth when credentials are configured."""
    if not settings.google_client_id or not settings.google_client_secret:
        return

    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url=(
            "https://accounts.google.com/.well-known/openid-configuration"
        ),
        client_kwargs={
            "scope": "openid email profile",
        },
    )


configure_google_oauth()

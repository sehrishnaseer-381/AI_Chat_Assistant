import logging

logger = logging.getLogger(__name__)
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.session import get_db
from app.models import User
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    GoogleTicketExchangeRequest,
)
from app.schemas.user import UserRead
from app.api.deps import get_current_user
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import RedirectResponse

from app.core.config import get_settings
from app.models import UserIdentity
from app.services.google_oauth_service import oauth

from fastapi.security import OAuth2PasswordRequestForm
import hashlib

from sqlalchemy import update

from app.models import GoogleLoginTicket

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
)
def register_user(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
):
    existing_user = db.scalar(
        select(User).where(User.email == payload.email)
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
    )

    db.add(user)

    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    return user


@router.post("/login", response_model=TokenResponse)
def login_user(
    payload: LoginRequest,
    db: Session = Depends(get_db),
):
    user = db.scalar(
        select(User).where(User.email == payload.email)
    )

    if (
        user is None
        or user.password_hash is None
        or not verify_password(
            payload.password,
            user.password_hash,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(user.id)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
    )



@router.post("/token", response_model=TokenResponse)
def login_for_swagger(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    # OAuth2 uses the field name "username"; we use email as its value.
    user = db.scalar(
        select(User).where(User.email == form_data.username)
    )

    if (
        user is None
        or user.password_hash is None
        or not verify_password(
            form_data.password,
            user.password_hash,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(
        access_token=create_access_token(user.id),
        token_type="bearer",
    )

@router.get("/me", response_model=UserRead)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user


@router.get("/google/login")
async def google_login(request: Request):
    """Start Google OAuth login."""
    settings = get_settings()

    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google login is not configured.",
        )

    redirect_uri = settings.google_redirect_uri

    return await oauth.google.authorize_redirect(
        request,
        redirect_uri,
    )


@router.get("/google/callback")
async def google_callback(
    request: Request,
    db: Session = Depends(get_db),
):
    """Handle Google's OAuth callback and issue a one-time login ticket."""
    settings = get_settings()


    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo")

        if not user_info:
            user_info = await oauth.google.userinfo(token=token)

    except Exception:
        logger.exception("Google OAuth callback failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google authentication failed. Please try again.",
        )


    if not user_info or user_info.get("email_verified") is not True:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A verified Google email is required.",
        )

    google_subject = user_info.get("sub")
    email = user_info.get("email")

    if not google_subject or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google did not provide the required account information.",
        )

    identity = db.scalar(
        select(UserIdentity).where(
            UserIdentity.provider == "google",
            UserIdentity.provider_user_id == google_subject,
        )
    )

    if identity:
        user = db.get(User, identity.user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="The linked account could not be found.",
            )
    else:
        user = db.scalar(
            select(User).where(User.email == email)
        )

        if user is None:
            user = User(email=email, password_hash=None)
            db.add(user)
            db.flush()

        identity = UserIdentity(
            user_id=user.id,
            provider="google",
            provider_user_id=google_subject,
        )
        db.add(identity)

    raw_ticket = secrets.token_urlsafe(32)
    ticket_hash = hashlib.sha256(raw_ticket.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)

    db.add(
        GoogleLoginTicket(
            ticket_hash=ticket_hash,
            user_id=user.id,
            expires_at=now + timedelta(minutes=2),
        )
    )

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not link this Google account. Please try again.",
        )

    frontend_url = (
        f"{settings.frontend_origin}/?"
        f"{urlencode({'google_ticket': raw_ticket})}"
    )

    response = RedirectResponse(url=frontend_url, status_code=303)
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/google/exchange", response_model=TokenResponse)
def exchange_google_ticket(
    payload: GoogleTicketExchangeRequest,
    db: Session = Depends(get_db),
):
    """Exchange a short-lived, single-use Google ticket for a JWT."""
    raw_ticket = payload.ticket

    if not isinstance(raw_ticket, str) or not raw_ticket:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid Google login ticket is required.",
        )

    ticket_hash = hashlib.sha256(raw_ticket.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)

    # Atomic consumption prevents the same ticket being used twice.
    statement = (
        update(GoogleLoginTicket)
        .where(
            GoogleLoginTicket.ticket_hash == ticket_hash,
            GoogleLoginTicket.consumed_at.is_(None),
            GoogleLoginTicket.expires_at > now,
        )
        .values(consumed_at=now)
        .returning(GoogleLoginTicket.user_id)
    )

    user_id = db.execute(statement).scalar_one_or_none()

    if user_id is None:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The Google login ticket is invalid, expired, or already used.",
        )

    user = db.get(User, user_id)

    if user is None:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The associated user account could not be found.",
        )

    db.commit()

    return TokenResponse(
        access_token=create_access_token(user.id),
        token_type="bearer",
    )

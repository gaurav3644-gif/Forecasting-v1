import threading
import time
import asyncio
import random
from fastapi import Body
from typing import Any, Dict, Optional
import os
import httpx
import smtplib
import socket
import ssl
from email.message import EmailMessage
from assistant_context_packet import build_context_packet
from assistant_openai import AssistantLLMError, answer_question_openai
import retriever
import hashlib
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from urllib.parse import quote, urlparse
import base64
import hmac
import hashlib as _hashlib
import pandas as pd
import plotly.graph_objects as go
from run_forecast2 import forecast_all_combined
import io
from datetime import datetime, timedelta, timezone
from supply_planner import (
    generate_supply_plan as generate_order_recommendations,
    generate_time_phased_supply_plan as generate_time_phased_supply_plan,
)
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


app = FastAPI()
# Use absolute path for templates to ensure it works in all deployment environments
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_auth_cookie_secret = (os.getenv("AUTH_COOKIE_SECRET") or "").strip()
if not _auth_cookie_secret:
    # For local/dev use only. In production, set AUTH_COOKIE_SECRET to a strong random value.
    _auth_cookie_secret = "dev-secret-change-me"
_auth_cookie_secret_b = _auth_cookie_secret.encode("utf-8")
_auth_cookie_name = (os.getenv("AUTH_COOKIE_NAME") or "").strip() or "fa_user"
_auth_cookie_max_age_s = int(float(os.getenv("AUTH_COOKIE_MAX_AGE_S", "0") or "0"))  # 0 => session cookie
_auth_cookie_force_secure = (os.getenv("AUTH_COOKIE_SECURE", "0") or "0").strip() == "1"

def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")

def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))

def _auth_cookie_value(email: str) -> str:
    email_b64 = _b64url_encode(email.encode("utf-8"))
    sig = hmac.new(_auth_cookie_secret_b, email_b64.encode("ascii"), digestmod=_hashlib.sha256).digest()
    return f"{email_b64}.{_b64url_encode(sig)}"

def _auth_cookie_email(cookie_val: str) -> Optional[str]:
    try:
        if not isinstance(cookie_val, str) or "." not in cookie_val:
            return None
        email_b64, sig_b64 = cookie_val.split(".", 1)
        expected_sig = hmac.new(_auth_cookie_secret_b, email_b64.encode("ascii"), digestmod=_hashlib.sha256).digest()
        actual_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        email = _b64url_decode(email_b64).decode("utf-8", errors="strict")
        return email
    except Exception:
        return None

def _get_user_email(request: Request) -> Optional[str]:
    try:
        return getattr(request.state, "user_email", None)
    except Exception:
        return None

def _is_signed_in(request: Request) -> bool:
    return bool(_get_user_email(request))

def _send_demo_request_email(
    *,
    name: str,
    email: str,
    company: Optional[str],
    phone: Optional[str],
    role: Optional[str],
    message: Optional[str],
    source_path: Optional[str],
    user_email: Optional[str],
) -> None:
    """
    Sends an email notification for a demo request using SMTP.

    Config (env vars):
      - PITENSOR_SMTP_HOST (required to send)
      - PITENSOR_SMTP_PORT (default 587)
      - PITENSOR_SMTP_USER (optional)
      - PITENSOR_SMTP_PASSWORD (optional)
      - PITENSOR_SMTP_FROM (optional; defaults to PITENSOR_SMTP_USER or support@pitensor.com)
      - PITENSOR_SMTP_USE_TLS (default 1)  -> STARTTLS for non-SSL
      - PITENSOR_SMTP_USE_SSL (default 0)  -> SMTP over SSL (usually 465)
      - PITENSOR_SMTP_TIMEOUT_S (default 12)
      - PITENSOR_DEMO_EMAIL_TO (default support@pitensor.com)
    """
    smtp_host = (os.getenv("PITENSOR_SMTP_HOST") or "").strip()
    if not smtp_host:
        raise RuntimeError("PITENSOR_SMTP_HOST is not set")

    to_addr = (os.getenv("PITENSOR_DEMO_EMAIL_TO") or "support@pitensor.com").strip() or "support@pitensor.com"

    smtp_port = int(float((os.getenv("PITENSOR_SMTP_PORT") or "587").strip() or "587"))
    smtp_user = (os.getenv("PITENSOR_SMTP_USER") or "").strip()
    smtp_password = (os.getenv("PITENSOR_SMTP_PASSWORD") or "").strip()
    smtp_from = (os.getenv("PITENSOR_SMTP_FROM") or "").strip() or (smtp_user or "support@pitensor.com")
    use_tls = (os.getenv("PITENSOR_SMTP_USE_TLS") or "1").strip() == "1"
    use_ssl = (os.getenv("PITENSOR_SMTP_USE_SSL") or "0").strip() == "1"
    timeout_s = float((os.getenv("PITENSOR_SMTP_TIMEOUT_S") or "12").strip() or "12")

    subject = f"[PiTensor] Request Demo: {name} ({email})"
    lines = [
        "New demo request received.",
        "",
        f"Name: {name}",
        f"Email: {email}",
        f"Company: {company or ''}",
        f"Role: {role or ''}",
        f"Phone/WhatsApp: {phone or ''}",
        f"Source: {source_path or ''}",
        f"Signed-in user: {user_email or ''}",
        "",
        "Message:",
        (message or "").strip(),
        "",
    ]
    body = "\n".join(lines)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = to_addr
    msg["Reply-To"] = email
    msg.set_content(body)

    context = ssl.create_default_context()

    def _send_via_smtp(*, host: str, port: int, ssl_mode: bool, tls_mode: bool) -> None:
        if ssl_mode:
            with smtplib.SMTP_SSL(host, port, timeout=timeout_s, context=context) as server:
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)
            return

        with smtplib.SMTP(host, port, timeout=timeout_s) as server:
            server.ehlo()
            if tls_mode:
                server.starttls(context=context)
                server.ehlo()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)

    try:
        _send_via_smtp(host=smtp_host, port=smtp_port, ssl_mode=use_ssl, tls_mode=use_tls)
        return
    except (TimeoutError, socket.timeout) as e:
        # Common on PaaS if outbound SMTP is filtered or slow to connect. Try SSL/465 if user didn't explicitly choose SSL.
        if not use_ssl and smtp_port == 587 and (os.getenv("PITENSOR_SMTP_USE_SSL") is None):
            try:
                _send_via_smtp(host=smtp_host, port=465, ssl_mode=True, tls_mode=False)
                return
            except Exception as e2:
                raise RuntimeError(
                    f"SMTP timed out on {smtp_host}:587 and fallback {smtp_host}:465 failed: {e2}"
                ) from e2
        raise e

def _send_demo_request_email_resend(
    *,
    name: str,
    email: str,
    company: Optional[str],
    phone: Optional[str],
    role: Optional[str],
    message: Optional[str],
    source_path: Optional[str],
    user_email: Optional[str],
) -> None:
    """
    Send demo request email via Resend (HTTPS API).

    Env vars:
      - PITENSOR_RESEND_API_KEY (required)
      - PITENSOR_DEMO_EMAIL_TO (default support@pitensor.com)
      - PITENSOR_EMAIL_FROM (default onboarding@resend.dev)
    """
    api_key = (os.getenv("PITENSOR_RESEND_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("PITENSOR_RESEND_API_KEY is not set")

    to_addr = (os.getenv("PITENSOR_DEMO_EMAIL_TO") or "support@pitensor.com").strip() or "support@pitensor.com"
    from_addr = (os.getenv("PITENSOR_EMAIL_FROM") or "").strip() or "onboarding@resend.dev"

    subject = f"[PiTensor] Request Demo: {name} ({email})"
    lines = [
        "New demo request received.",
        "",
        f"Name: {name}",
        f"Email: {email}",
        f"Company: {company or ''}",
        f"Role: {role or ''}",
        f"Phone/WhatsApp: {phone or ''}",
        f"Source: {source_path or ''}",
        f"Signed-in user: {user_email or ''}",
        "",
        "Message:",
        (message or "").strip(),
        "",
    ]
    body = "\n".join(lines)

    timeout_s = float((os.getenv("PITENSOR_EMAIL_TIMEOUT_S") or os.getenv("PITENSOR_SMTP_TIMEOUT_S") or "12").strip() or "12")
    resp = httpx.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "from": from_addr,
            "to": [to_addr],
            "subject": subject,
            "text": body,
            "reply_to": email,
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()

def _send_demo_request_email_sendgrid(
    *,
    name: str,
    email: str,
    company: Optional[str],
    phone: Optional[str],
    role: Optional[str],
    message: Optional[str],
    source_path: Optional[str],
    user_email: Optional[str],
) -> None:
    """
    Send demo request email via SendGrid (HTTPS API).

    Env vars:
      - PITENSOR_SENDGRID_API_KEY (required)
      - PITENSOR_DEMO_EMAIL_TO (default support@pitensor.com)
      - PITENSOR_EMAIL_FROM (default support@pitensor.com)
    """
    api_key = (os.getenv("PITENSOR_SENDGRID_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("PITENSOR_SENDGRID_API_KEY is not set")

    to_addr = (os.getenv("PITENSOR_DEMO_EMAIL_TO") or "support@pitensor.com").strip() or "support@pitensor.com"
    from_addr = (os.getenv("PITENSOR_EMAIL_FROM") or "support@pitensor.com").strip() or "support@pitensor.com"

    subject = f"[PiTensor] Request Demo: {name} ({email})"
    lines = [
        "New demo request received.",
        "",
        f"Name: {name}",
        f"Email: {email}",
        f"Company: {company or ''}",
        f"Role: {role or ''}",
        f"Phone/WhatsApp: {phone or ''}",
        f"Source: {source_path or ''}",
        f"Signed-in user: {user_email or ''}",
        "",
        "Message:",
        (message or "").strip(),
        "",
    ]
    body = "\n".join(lines)

    timeout_s = float((os.getenv("PITENSOR_EMAIL_TIMEOUT_S") or os.getenv("PITENSOR_SMTP_TIMEOUT_S") or "12").strip() or "12")
    resp = httpx.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "personalizations": [{"to": [{"email": to_addr}]}],
            "from": {"email": from_addr},
            "reply_to": {"email": email},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()


def _send_text_email(
    *,
    to_addr: str,
    subject: str,
    body: str,
    reply_to: Optional[str] = None,
) -> str:
    """
    Send a plain-text email using the configured provider.

    Providers (env vars):
      - Resend: PITENSOR_RESEND_API_KEY (+ PITENSOR_EMAIL_FROM)
      - SendGrid: PITENSOR_SENDGRID_API_KEY (+ PITENSOR_EMAIL_FROM)
      - SMTP: PITENSOR_SMTP_HOST (+ PITENSOR_SMTP_USER/PASSWORD etc.)
    Optional:
      - PITENSOR_EMAIL_PROVIDER = resend|sendgrid|smtp
      - PITENSOR_EMAIL_TIMEOUT_S (fallback PITENSOR_SMTP_TIMEOUT_S)
    Returns provider name used.
    """
    to_addr = (to_addr or "").strip()
    if "@" not in to_addr:
        raise ValueError("valid to_addr is required")
    subject = (subject or "").strip() or "PiTensor"
    body = (body or "").strip()

    provider = (os.getenv("PITENSOR_EMAIL_PROVIDER") or "").strip().lower()
    has_resend = bool((os.getenv("PITENSOR_RESEND_API_KEY") or "").strip())
    has_sendgrid = bool((os.getenv("PITENSOR_SENDGRID_API_KEY") or "").strip())
    has_smtp = bool((os.getenv("PITENSOR_SMTP_HOST") or "").strip())
    timeout_s = float((os.getenv("PITENSOR_EMAIL_TIMEOUT_S") or os.getenv("PITENSOR_SMTP_TIMEOUT_S") or "12").strip() or "12")

    def _from_addr_default() -> str:
        return (os.getenv("PITENSOR_EMAIL_FROM") or "").strip() or "onboarding@resend.dev"

    def _send_resend() -> None:
        api_key = (os.getenv("PITENSOR_RESEND_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("PITENSOR_RESEND_API_KEY is not set")
        from_addr = _from_addr_default()
        payload = {
            "from": from_addr,
            "to": [to_addr],
            "subject": subject,
            "text": body,
        }
        if reply_to:
            payload["reply_to"] = reply_to
        resp = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()

    def _send_sendgrid() -> None:
        api_key = (os.getenv("PITENSOR_SENDGRID_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("PITENSOR_SENDGRID_API_KEY is not set")
        from_addr = (os.getenv("PITENSOR_EMAIL_FROM") or "").strip() or "support@pitensor.com"
        payload = {
            "personalizations": [{"to": [{"email": to_addr}]}],
            "from": {"email": from_addr},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
        }
        if reply_to:
            payload["reply_to"] = {"email": reply_to}
        resp = httpx.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()

    def _send_smtp() -> None:
        smtp_host = (os.getenv("PITENSOR_SMTP_HOST") or "").strip()
        if not smtp_host:
            raise RuntimeError("PITENSOR_SMTP_HOST is not set")
        smtp_port = int(float((os.getenv("PITENSOR_SMTP_PORT") or "587").strip() or "587"))
        smtp_user = (os.getenv("PITENSOR_SMTP_USER") or "").strip()
        smtp_password = (os.getenv("PITENSOR_SMTP_PASSWORD") or "").strip()
        smtp_from = (os.getenv("PITENSOR_SMTP_FROM") or "").strip() or (smtp_user or "support@pitensor.com")
        use_tls = (os.getenv("PITENSOR_SMTP_USE_TLS") or "1").strip() == "1"
        use_ssl = (os.getenv("PITENSOR_SMTP_USE_SSL") or "0").strip() == "1"

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = to_addr
        if reply_to:
            msg["Reply-To"] = reply_to
        msg.set_content(body)

        context = ssl.create_default_context()
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout_s, context=context) as server:
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout_s) as server:
                server.ehlo()
                if use_tls:
                    server.starttls(context=context)
                    server.ehlo()
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)

    def _do(p: str) -> None:
        if p == "resend":
            _send_resend()
        elif p == "sendgrid":
            _send_sendgrid()
        elif p == "smtp":
            _send_smtp()
        else:
            raise RuntimeError(f"Unknown PITENSOR_EMAIL_PROVIDER={p}")

    if provider:
        _do(provider)
        return provider

    if has_resend:
        _do("resend")
        return "resend"
    if has_sendgrid:
        _do("sendgrid")
        return "sendgrid"
    if has_smtp:
        _do("smtp")
        return "smtp"
    raise RuntimeError("No email provider configured")


def _auth_mode() -> str:
    mode = (os.getenv("PITENSOR_AUTH_MODE") or "").strip().lower()
    if mode in ("otp", "email_otp", "emailotp"):
        return "otp"
    if mode in ("simple", "email_only", "emailonly", "none"):
        return "email_only"
    # Auto: enable OTP when an email provider is configured.
    if (os.getenv("PITENSOR_RESEND_API_KEY") or "").strip():
        return "otp"
    if (os.getenv("PITENSOR_SENDGRID_API_KEY") or "").strip():
        return "otp"
    if (os.getenv("PITENSOR_SMTP_HOST") or "").strip():
        return "otp"
    return "email_only"


def _otp_hash(otp_id: str, code: str) -> str:
    val = f"{(otp_id or '').strip()}:{(code or '').strip()}".encode("utf-8")
    return hmac.new(_auth_cookie_secret_b, val, digestmod=_hashlib.sha256).hexdigest()


def _client_ip(request: Request) -> str:
    try:
        xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        return xff or (request.client.host if request.client else "")
    except Exception:
        return ""

def _user_friendly_email_send_error(e: Exception) -> str:
    try:
        if isinstance(e, httpx.HTTPStatusError) and getattr(e, "response", None) is not None:
            status = int(getattr(e.response, "status_code", 0) or 0)
            body = ""
            try:
                body = (e.response.text or "")[:1000]
            except Exception:
                body = ""

            if status == 403 and "You can only send testing emails to your own email address" in body:
                return (
                    "Email provider is in testing mode. "
                    "Verify your domain in Resend (Domains → pitensor.com) and set PITENSOR_EMAIL_FROM to an @pitensor.com address. "
                    "Until then, OTP emails can only be sent to your Resend account email."
                )

            # Generic status with short details.
            short = body
            try:
                short = short.replace("\n", " ").strip()
            except Exception:
                pass
            if short:
                return f"Email send failed (HTTP {status}): {short}"
            return f"Email send failed (HTTP {status})."
    except Exception:
        pass
    return str(e) or "Email send failed."

def _send_demo_request_notification(
    *,
    name: str,
    email: str,
    company: Optional[str],
    phone: Optional[str],
    role: Optional[str],
    message: Optional[str],
    source_path: Optional[str],
    user_email: Optional[str],
) -> str:
    """
    Send demo request notification using the best configured provider.
    Returns provider name used.
    """
    provider = (os.getenv("PITENSOR_EMAIL_PROVIDER") or "").strip().lower()
    has_resend = bool((os.getenv("PITENSOR_RESEND_API_KEY") or "").strip())
    has_sendgrid = bool((os.getenv("PITENSOR_SENDGRID_API_KEY") or "").strip())
    has_smtp = bool((os.getenv("PITENSOR_SMTP_HOST") or "").strip())

    def _do(p: str) -> None:
        if p == "resend":
            _send_demo_request_email_resend(
                name=name,
                email=email,
                company=company,
                phone=phone,
                role=role,
                message=message,
                source_path=source_path,
                user_email=user_email,
            )
        elif p == "sendgrid":
            _send_demo_request_email_sendgrid(
                name=name,
                email=email,
                company=company,
                phone=phone,
                role=role,
                message=message,
                source_path=source_path,
                user_email=user_email,
            )
        elif p == "smtp":
            _send_demo_request_email(
                name=name,
                email=email,
                company=company,
                phone=phone,
                role=role,
                message=message,
                source_path=source_path,
                user_email=user_email,
            )
        else:
            raise RuntimeError(f"Unknown PITENSOR_EMAIL_PROVIDER={p}")

    if provider:
        _do(provider)
        return provider

    # Auto: prefer HTTPS APIs (more likely to work on Railway) then SMTP.
    if has_resend:
        _do("resend")
        return "resend"
    if has_sendgrid:
        _do("sendgrid")
        return "sendgrid"
    if has_smtp:
        _do("smtp")
        return "smtp"
    raise RuntimeError("No email provider configured (set PITENSOR_RESEND_API_KEY or PITENSOR_SENDGRID_API_KEY or PITENSOR_SMTP_HOST)")

def _session_id_from_request(request: Request) -> str:
    """
    Session key for per-user data isolation.

    Note: Current "sign-in" is email-only (no verification). This isolates state between
    users in a running server, but is not strong authentication.
    """
    email = (_get_user_email(request) or "").strip().lower()
    return f"user:{email}" if email else "default"

def _safe_next_path(next_val: Optional[str]) -> str:
    dest = (next_val or "").strip()
    if not dest:
        return "/"
    parsed = urlparse(dest)
    if parsed.scheme or parsed.netloc:
        return "/"
    if not dest.startswith("/") or dest.startswith("//"):
        return "/"
    return dest

def _cookie_secure(request: Request) -> bool:
    if _auth_cookie_force_secure:
        return True
    try:
        return (request.url.scheme or "").lower() == "https"
    except Exception:
        return False

def _humanize_feature_name(feature: str) -> str:
    f = str(feature or "")
    if f.startswith("lag_"):
        days = f.replace("lag_", "")
        return f"Sales {days}d ago"
    if f.startswith("rolling_mean_"):
        d = f.replace("rolling_mean_", "")
        return f"Rolling mean ({d}d)"
    if f.startswith("rolling_std_"):
        d = f.replace("rolling_std_", "")
        return f"Rolling std dev ({d}d)"
    if f in {"dow", "month"}:
        return "Day of week" if f == "dow" else "Month"
    if f == "is_q4":
        return "Q4 seasonality"
    if f == "months_to_dec":
        return "Months to December"
    if f == "is_pre_peak":
        return "Pre-peak season"
    if f == "is_peak_build":
        return "Peak build (Nov)"
    if f == "time_idx":
        return "Time trend"
    if f == "time_x_is_peak":
        return "Trend × peak build"
    if f == "sin_y":
        return "Yearly seasonality (sin)"
    if f == "cos_y":
        return "Yearly seasonality (cos)"
    known = {
        "price": "Price",
        "promo": "Promotion",
        "promo_discount": "Promo discount",
        "out_of_stock": "Out of stock",
        "inventory_level": "Inventory level",
        "holiday": "Holiday",
        "weather_index": "Weather index",
        "competitor_price": "Competitor price",
    }
    if f in known:
        return known[f]
    if "_" in f:
        prefix, rest = f.split("_", 1)
        if prefix not in {"lag", "rolling", "time", "sin", "cos"}:
            return f"{prefix} = {rest}"
    return f

@app.middleware("http")
async def _optional_auth_guard(request: Request, call_next):
    """
    Optional sign-in enforcement (disabled by default).
    Set REQUIRE_AUTH=1 to require an email session for most UI routes.
    """
    # Populate user context from cookie for templates/handlers.
    request.state.user_email = _auth_cookie_email(request.cookies.get(_auth_cookie_name, "")) if hasattr(request, "cookies") else None

    require_auth = (os.getenv("REQUIRE_AUTH", "0") or "0").strip() == "1"
    if not require_auth:
        return await call_next(request)

    path = request.url.path or "/"
    # Allow public routes
    allow_prefixes = ("/signin", "/signup", "/signout", "/auth/", "/static", "/docs", "/openapi.json", "/")
    if any(path == p for p in ("/",)) or any(path.startswith(p) for p in allow_prefixes if p != "/"):
        return await call_next(request)

    # Allow health/status endpoints
    if path in ("/status", "/forecast_status"):
        return await call_next(request)

    if _is_signed_in(request):
        return await call_next(request)

    next_q = quote(path)
    return RedirectResponse(f"/signin?next={next_q}", status_code=303)

# Import and register connectors router
from connectors import router as connectors_router
app.include_router(connectors_router)

# Decision Intelligence routes (generate/score options)
try:
    from decision_routes import router as decision_router
    app.include_router(decision_router)
except Exception as _e:
    # Keep the app running even if decision modules are missing in some deployments.
    print(f"[WARN] decision_routes not loaded: {_e}")
# Global storage for simplicity (use database for production SaaS)
# Global storage for simplicity (use database for production SaaS)
data_store = {}
app.state.data_store = data_store

_otp_mem_lock = threading.Lock()
_otp_mem_store: dict[str, dict[str, Any]] = {}

def _otp_mem_put(*, otp_id: str, email: str, code_hash: str, expires_at: str) -> None:
    with _otp_mem_lock:
        _otp_mem_store[otp_id] = {
            "id": otp_id,
            "email": email,
            "code_hash": code_hash,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "expires_at": expires_at,
            "attempts": 0,
            "consumed_at": None,
        }

def _otp_mem_get(*, otp_id: str) -> Optional[Dict[str, Any]]:
    with _otp_mem_lock:
        rec = _otp_mem_store.get(otp_id)
        return dict(rec) if rec else None

def _otp_mem_inc_attempts(*, otp_id: str) -> None:
    with _otp_mem_lock:
        rec = _otp_mem_store.get(otp_id)
        if rec:
            rec["attempts"] = int(rec.get("attempts") or 0) + 1

def _otp_mem_consume(*, otp_id: str) -> None:
    with _otp_mem_lock:
        rec = _otp_mem_store.get(otp_id)
        if rec:
            rec["consumed_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

_assistant_locks: dict[str, asyncio.Lock] = {}
_assistant_cache: dict[str, tuple[float, str]] = {}
_assistant_cache_ttl_s: float = float(os.getenv("AI_ASSISTANT_CACHE_TTL_S", "60"))
_assistant_debug_enabled: bool = (os.getenv("AI_ASSISTANT_DEBUG", "0") or "0").strip() == "1"

def _assistant_debug(msg: str) -> None:
    if not _assistant_debug_enabled:
        return
    try:
        ts = datetime.utcnow().isoformat(timespec="seconds")
    except Exception:
        ts = "unknown-time"
    print(f"[ASSISTANT {ts}Z] {msg}")

_SUPPLY_PLAN_GLOSSARY: dict[str, str] = {
    "forecast_demand": "Forecasted demand for the month (monthly bucket).",
    "beginning_on_hand": "On-hand at the start of the month after adding receipts arriving that month.",
    "receipts": "Quantity arriving this month from orders placed in prior months (after lead time).",
    "order_qty": "Order placed this month (arrives later after lead time).",
    "ending_on_hand": "On-hand after consuming forecast_demand for the month (0 if stockout).",
    "inventory_position": "On-hand plus on-order pipeline: on_hand + sum(pipeline).",
    "lead_time_days": "Supplier lead time input in days.",
    "lead_time_months": "ceil(lead_time_days/30) (min 1). Used by monthly simulation.",
    "safety_stock": "Buffer stock based on service_level and demand variability, scaled by sqrt(lead_time_months).",
    "reorder_point": "safety_stock + lead_time_demand (lead-time demand is next lead_time_months of forecast).",
    "target_level": "Order-up-to target: safety_stock + cover_demand (next lead_time_months+1 months).",
    "service_level": "Target in-stock probability (0–1). Higher -> more safety stock.",
    "moq": "Minimum Order Quantity (minimum order when ordering).",
    "order_multiple": "Order quantity rounded up to this multiple.",
    "max_capacity_per_week": "Optional capacity per week, approximated to monthly as *4.",
    "input_on_hand": "Raw inventory snapshot input (before subtracting allocations/backorders).",
    "input_allocated": "Units reserved/committed elsewhere (reduces availability).",
    "input_backorders": "Units already owed (reduces availability).",
    "starting_net_on_hand": "max(0, on_hand - allocated - backorders): usable starting inventory.",
    "stockout_qty": "Unmet demand for the month (if any).",
    "risk_flag": "OK or STOCKOUT depending on whether the month stocked out.",
}

def _cache_get(key: str) -> Optional[str]:
    row = _assistant_cache.get(key)
    if not row:
        return None
    ts, val = row
    if (time.time() - ts) > _assistant_cache_ttl_s:
        _assistant_cache.pop(key, None)
        return None
    return val

def _cache_set(key: str, val: str) -> None:
    _assistant_cache[key] = (time.time(), val)
    # Simple cap to avoid unbounded growth
    if len(_assistant_cache) > 500:
        # Drop oldest ~20%
        items = sorted(_assistant_cache.items(), key=lambda kv: kv[1][0])
        for k, _ in items[:100]:
            _assistant_cache.pop(k, None)

def _get_lock(session_id: str) -> asyncio.Lock:
    lock = _assistant_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _assistant_locks[session_id] = lock
    return lock

def _safe_text(x: object, max_len: int = 6000) -> str:
    s = str(x) if x is not None else ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if len(s) > max_len:
        return s[:max_len] + "\n...(truncated)"
    return s

def _http_error_detail(e: Exception) -> str:
    if not isinstance(e, httpx.HTTPStatusError):
        return str(e)
    try:
        data = e.response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err.get("message"))
            if data.get("message"):
                return str(data.get("message"))
        return (e.response.text or "").strip()[:800]
    except Exception:
        return (getattr(e.response, "text", "") or "").strip()[:800] or str(e)

def _llm_unavailable_prefix(provider: str, e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        detail = _http_error_detail(e)
        if provider in ("huggingface", "hf") and status == 404:
            return (
                f"LLM error ({provider}): HTTP 404 (endpoint not found).\n"
                f"Details: {detail}\n"
                "Fix: set HUGGINGFACE_BASE_URL=https://router.huggingface.co (and ensure you have token permissions for Inference Providers)."
            )
        if provider in ("huggingface", "hf") and status == 400:
            # HF Router often returns this when the requested model isn't available for the enabled providers on the account.
            return (
                f"LLM error ({provider}): HTTP 400 (bad request).\n"
                f"Details: {detail}\n"
                "Fix: pick a model supported by your enabled Hugging Face Inference Providers (or enable a provider that supports it), "
                "then set HUGGINGFACE_MODEL accordingly. If you want a no-quota option, use AI_ASSISTANT_PROVIDER=ollama (local)."
            )
        if status == 429:
            return (
                f"LLM error ({provider}): 429 Too Many Requests.\n"
                f"Details: {detail}\n"
                "Likely causes: (1) quota/billing not enabled, (2) RPM/TPM limit reached, or (3) duplicate/parallel requests. "
                "Try waiting 30–60s, then retry once; or set AI_ASSISTANT_DEBUG=1 and check server logs."
            )
        if status in (401, 403):
            if provider in ("huggingface", "hf"):
                return (
                    f"LLM error ({provider}): {status} (auth/permission).\n"
                    f"Details: {detail}\n"
                    "Fix: create a Hugging Face user access token with Inference permissions (or enable Inference Providers/billing on your account), "
                    "then set HF_TOKEN/HUGGINGFACE_API_KEY. If you want truly free/no-quotas, use provider=ollama (local)."
                )
            return f"LLM error ({provider}): {status} (auth/permission). Details: {detail}"
        return f"LLM error ({provider}): HTTP {status}. Details: {detail}"
    detail = str(e).strip()
    if not detail:
        detail = type(e).__name__
    if provider == "ollama" and isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
        return (
            f"LLM error ({provider}): {detail}\n"
            "Fix: ensure Ollama is running and reachable at OLLAMA_BASE_URL (default: http://localhost:11434). "
            "Test: `curl http://localhost:11434/api/tags`."
        )
    return f"LLM error ({provider}): {detail}"

def _fmt_num(x: float | int | None) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
        if not pd.notna(v):
            return "n/a"
        if abs(v) >= 1_000_000:
            return f"{v:,.0f}"
        if abs(v) >= 1_000:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.0f}"
        return f"{v:.2f}"
    except Exception:
        return str(x)

def _date_range_str(df: pd.DataFrame, date_col: str = "date") -> str | None:
    if date_col not in df.columns:
        return None
    d = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if d.empty:
        return None
    return f"{d.min().date().isoformat()} to {d.max().date().isoformat()}"

def _top_n_pairs(df: pd.DataFrame, group_col: str, value_col: str, n: int = 3) -> list[tuple[str, float]]:
    if group_col not in df.columns or value_col not in df.columns:
        return []
    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[group_col, value_col])
    if work.empty:
        return []
    top = work.groupby(group_col, as_index=True)[value_col].sum().sort_values(ascending=False).head(n)
    return [(str(idx), float(val)) for idx, val in top.items()]

def _summarize_raw_data(df: pd.DataFrame) -> str:
    date_range = _date_range_str(df, "date") or "n/a"
    sales_total = None
    if "sales" in df.columns:
        sales_total = float(pd.to_numeric(df["sales"], errors="coerce").fillna(0.0).sum())
    top_items = _top_n_pairs(df, "item", "sales", 3)
    top_stores = _top_n_pairs(df, "store", "sales", 3)
    missing = df.isna().sum().to_dict()
    missing_key = ", ".join(f"{k}={int(v)}" for k, v in list(missing.items())[:6])

    parts = [
        f"Raw data: {df.shape[0]:,} rows × {df.shape[1]:,} columns.",
        f"Date range: {date_range}.",
    ]
    if sales_total is not None:
        parts.append(f"Total sales: {_fmt_num(sales_total)}.")
    if top_items:
        parts.append("Top items: " + ", ".join(f"{k} ({_fmt_num(v)})" for k, v in top_items) + ".")
    if top_stores:
        parts.append("Top stores: " + ", ".join(f"{k} ({_fmt_num(v)})" for k, v in top_stores) + ".")
    parts.append(f"Missing values (sample): {missing_key}.")
    return " ".join(parts)

def _summarize_forecast(forecast_df: pd.DataFrame) -> str:
    work = forecast_df.copy()
    if "forecast" not in work.columns:
        quantile_fallback = next((c for c in ["forecast_p60", "forecast_p50"] if c in work.columns), None)
        if quantile_fallback:
            work = work.rename(columns={quantile_fallback: "forecast"})

    date_range = _date_range_str(work, "date") or "n/a"
    total_forecast = None
    if "forecast" in work.columns:
        total_forecast = float(pd.to_numeric(work["forecast"], errors="coerce").fillna(0.0).sum())

    top_month_str = None
    if "date" in work.columns and "forecast" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work["forecast"] = pd.to_numeric(work["forecast"], errors="coerce")
        month_sum = work.dropna(subset=["date"]).groupby(work["date"].dt.to_period("M"), as_index=True)["forecast"].sum()
        if not month_sum.empty:
            top_month = month_sum.idxmax()
            top_month_str = f"{top_month.strftime('%B %Y')} ({_fmt_num(float(month_sum.max()))})"

    top_items = _top_n_pairs(work, "item", "forecast", 3)
    top_stores = _top_n_pairs(work, "store", "forecast", 3)

    parts = [
        f"Forecast: {work.shape[0]:,} rows.",
        f"Date range: {date_range}.",
    ]
    if total_forecast is not None:
        parts.append(f"Total forecast: {_fmt_num(total_forecast)}.")
    if top_month_str:
        parts.append(f"Top forecast month: {top_month_str}.")
    if top_items:
        parts.append("Top items: " + ", ".join(f"{k} ({_fmt_num(v)})" for k, v in top_items) + ".")
    if top_stores:
        parts.append("Top stores: " + ", ".join(f"{k} ({_fmt_num(v)})" for k, v in top_stores) + ".")
    return " ".join(parts)

def _summarize_supply_plan(plan_df: pd.DataFrame) -> str:
    work = plan_df.copy()
    parts = [f"Supply plan: {work.shape[0]:,} rows × {work.shape[1]:,} columns."]

    if "period_start" in work.columns:
        ps = pd.to_datetime(work["period_start"], errors="coerce").dropna()
        if not ps.empty:
            parts.append(f"Period range: {ps.min().date().isoformat()} to {ps.max().date().isoformat()}.")

    if "order_qty" in work.columns:
        total_orders = float(pd.to_numeric(work["order_qty"], errors="coerce").fillna(0.0).sum())
        parts.append(f"Total ordered: {_fmt_num(total_orders)}.")

    if "stockout_qty" in work.columns:
        total_stockout = float(pd.to_numeric(work["stockout_qty"], errors="coerce").fillna(0.0).sum())
        parts.append(f"Total stockout: {_fmt_num(total_stockout)}.")

    if "risk_flag" in work.columns:
        stockout_rows = int((work["risk_flag"].astype(str) == "STOCKOUT").sum())
        parts.append(f"Rows flagged STOCKOUT: {stockout_rows:,}.")

    # Try to show top ordered items/stores if available.
    if "order_qty" in work.columns:
        item_col = "item" if "item" in work.columns else None
        store_col = "store" if "store" in work.columns else None
        if item_col:
            top_items = _top_n_pairs(work, item_col, "order_qty", 3)
            if top_items:
                parts.append("Top items by order: " + ", ".join(f"{k} ({_fmt_num(v)})" for k, v in top_items) + ".")
        if store_col:
            top_stores = _top_n_pairs(work, store_col, "order_qty", 3)
            if top_stores:
                parts.append("Top stores by order: " + ", ".join(f"{k} ({_fmt_num(v)})" for k, v in top_stores) + ".")

    return " ".join(parts)

def _compact_preview(df: pd.DataFrame, max_rows: int = 5, max_cols: int = 10) -> str:
    try:
        cols = [str(c) for c in df.columns[:max_cols]]
        view = df[cols].head(max_rows)
        return view.to_string(index=False)
    except Exception:
        return ""

def _build_assistant_context(session: dict, *, include_preview: bool = False) -> str:
    raw_df = session.get("df")
    forecast_df = session.get("forecast_df")
    supply_plan_df = session.get("supply_plan_df")

    blocks: list[str] = []
    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        blocks.append("RAW_DATA_SUMMARY:\n" + _summarize_raw_data(raw_df))
        if include_preview:
            prev = _compact_preview(raw_df, max_rows=5, max_cols=8)
            if prev:
                blocks.append("RAW_DATA_PREVIEW (first rows):\n" + prev)

    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        blocks.append("FORECAST_SUMMARY:\n" + _summarize_forecast(forecast_df))
        if include_preview:
            prev = _compact_preview(forecast_df, max_rows=5, max_cols=10)
            if prev:
                blocks.append("FORECAST_PREVIEW (first rows):\n" + prev)

    if isinstance(supply_plan_df, pd.DataFrame) and not supply_plan_df.empty:
        blocks.append("SUPPLY_PLAN_SUMMARY:\n" + _summarize_supply_plan(supply_plan_df))
        if include_preview:
            prev = _compact_preview(supply_plan_df, max_rows=6, max_cols=12)
            if prev:
                blocks.append("SUPPLY_PLAN_PREVIEW (first rows):\n" + prev)

    if not blocks:
        return "No data loaded yet in this session."

    return "\n\n".join(blocks)


def _build_llm_context_packet(session: dict, *, combo_key: Optional[str] = None) -> dict[str, Any]:
    """
    Canonical data blocks used for Q&A:
      1) Raw Sales (facts)
      2) Forecast Output
      3) Supply Plan & Risks

    The LLM is instructed to answer ONLY from this packet.
    """
    packet = build_context_packet(session, combo_key=combo_key)
    try:
        ctx = build_planning_context(session, combo_key=combo_key)
        risks = detect_risks(session, context=ctx)
        actions = generate_actions(session, context=ctx, risks=risks)
    except Exception:
        ctx = {"selected_combo_key": combo_key}
        risks = []
        actions = []

    packet["supply_plan_and_risks"] = {
        "planning_context": ctx,
        "risks": risks,
        "actions": actions,
    }
    return packet

async def _openai_chat(user_message: str, context: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "400"))
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    timeout_s = float(os.getenv("OPENAI_TIMEOUT_S", "30"))

    system_prompt = (
        "You are PiTensor, an expert assistant for interpreting uploaded sales data, forecast outputs, and supply planning outputs. "
        "Use the provided context only. If the answer is not in the context, say what extra info you need.\n\n"
        f"CONTEXT:\n{_safe_text(context)}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        _assistant_debug(f"OpenAI request: model={model} max_tokens={max_tokens} temperature={temperature}")
        res = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        try:
            res.raise_for_status()
        except httpx.HTTPStatusError as e:
            _assistant_debug(f"OpenAI HTTP {e.response.status_code}: {_http_error_detail(e)}")
            raise
        data = res.json()
        return str(data["choices"][0]["message"]["content"]).strip()

async def _gemini_chat(user_message: str, context: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "400"))
    temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

    system_text = (
        "You are PiTensor, an expert assistant for interpreting uploaded sales data, forecast outputs, and supply planning outputs. "
        "Use the provided context only. If the answer is not in the context, say what extra info you need."
    )
    # Keep the schema conservative (single user content); works even if system_instruction is not supported.
    prompt = f"{system_text}\n\nCONTEXT:\n{_safe_text(context)}\n\nUSER_QUESTION:\n{user_message}"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        _assistant_debug(f"Gemini request: model={model} max_tokens={max_tokens} temperature={temperature}")
        res = await client.post(url, params={"key": api_key}, json=body)
        try:
            res.raise_for_status()
        except httpx.HTTPStatusError as e:
            _assistant_debug(f"Gemini HTTP {e.response.status_code}: {_http_error_detail(e)}")
            raise
        data = res.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return "No response from Gemini."
        content = (candidates[0].get("content") or {})
        parts = content.get("parts") or []
        text = parts[0].get("text") if parts else None
        return str(text or "").strip() or "No text response from Gemini."

async def _ollama_chat(user_message: str, context: str) -> str:
    """
    Local LLM via Ollama (no per-request API cost).
    Requires Ollama running locally (default: http://localhost:11434).
    """
    base_url = (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") or "").strip().rstrip("/")
    model = (os.getenv("OLLAMA_MODEL", "mistral:latest") or "").strip()
    if not base_url:
        raise ValueError("OLLAMA_BASE_URL is empty.")
    if not model:
        raise ValueError("OLLAMA_MODEL is empty.")

    max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "400"))
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT_S", "180"))

    system_prompt = (
        "You are PiTensor, an expert assistant for interpreting uploaded sales data, forecast outputs, and supply planning outputs. "
        "Use the provided context only. If the answer is not in the context, say what extra info you need.\n\n"
        f"CONTEXT:\n{_safe_text(context)}"
    )

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "options": {
            "temperature": temperature,
            # Ollama doesn't guarantee OpenAI-style max_tokens, but num_predict works similarly.
            "num_predict": max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        _assistant_debug(f"Ollama request: base_url={base_url} model={model} max_tokens={max_tokens} temperature={temperature}")
        res = await client.post(f"{base_url}/api/chat", json=payload)
        try:
            res.raise_for_status()
        except httpx.HTTPStatusError as e:
            _assistant_debug(f"Ollama HTTP {e.response.status_code}: {_http_error_detail(e)}")
            raise
        data = res.json()
        msg = data.get("message") or {}
        content = msg.get("content")
        return str(content or "").strip() or "No response from local model."

async def _huggingface_chat(user_message: str, context: str) -> str:
    """
    Hugging Face Inference Providers via router (OpenAI-compatible chat completions).

    Env:
      - HUGGINGFACE_API_KEY or HF_API_KEY or HF_TOKEN
      - HUGGINGFACE_MODEL (e.g. "mistralai/Mistral-7B-Instruct-v0.2")
      - HUGGINGFACE_MAX_TOKENS (default: 400)
      - HUGGINGFACE_TEMPERATURE (default: 0.2)
      - HUGGINGFACE_BASE_URL (default: "https://router.huggingface.co")
    """
    api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY (or HF_API_KEY / HF_TOKEN) is not set.")

    model = (os.getenv("HUGGINGFACE_MODEL") or "mistralai/Mistral-7B-Instruct-v0.2").strip()

    max_tokens = int(os.getenv("HUGGINGFACE_MAX_TOKENS", "400"))
    temperature = float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.2"))

    base_url = (os.getenv("HUGGINGFACE_BASE_URL") or "https://router.huggingface.co").strip().rstrip("/")
    # HF deprecated api-inference.huggingface.co for providers; auto-upgrade if configured.
    if "api-inference.huggingface.co" in base_url:
        base_url = "https://router.huggingface.co"

    system_prompt = (
        "You are PiTensor, an expert assistant for interpreting uploaded sales data, forecast outputs, and supply planning outputs. "
        "Use the provided context only. If the answer is not in the context, say what extra info you need."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"{system_prompt}\n\nCONTEXT:\n{_safe_text(context)}"},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    candidate_urls = [
        # Router OpenAI-compatible endpoint
        f"{base_url}/v1/chat/completions",
        # Some deployments use an explicit provider namespace
        f"{base_url}/hf-inference/v1/chat/completions",
        # Some deployments include model in the path
        f"{base_url}/hf-inference/models/{model}/v1/chat/completions",
        f"{base_url}/models/{model}/v1/chat/completions",
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        last_http_err: httpx.HTTPStatusError | None = None
        for url in candidate_urls:
            _assistant_debug(f"HF request: url={url} model={model} max_tokens={max_tokens} temperature={temperature}")
            res = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            if res.status_code in (404, 405):
                # Try next candidate route.
                try:
                    res.raise_for_status()
                except httpx.HTTPStatusError as e:
                    last_http_err = e
                    _assistant_debug(f"HF route not found: HTTP {e.response.status_code} for {url}")
                    continue

            try:
                res.raise_for_status()
            except httpx.HTTPStatusError as e:
                _assistant_debug(f"HF HTTP {e.response.status_code}: {_http_error_detail(e)}")
                # 404 on the router can be an endpoint mismatch; try next candidate.
                if e.response.status_code in (404, 405):
                    last_http_err = e
                    continue
                raise

            data = res.json()
            return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip() or "No response from Hugging Face."

        if last_http_err is not None:
            raise last_http_err
        raise RuntimeError("Hugging Face request failed: no valid endpoint matched.")

def _default_provider() -> str:
    p = (os.getenv("AI_ASSISTANT_PROVIDER") or "").strip().lower()
    if p:
        return p
    # Prefer OpenAI if configured; otherwise use Ollama if present, then other providers.
    # Override explicitly via AI_ASSISTANT_PROVIDER=openai|ollama|gemini|huggingface|local.
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_BASE_URL"):
        return "ollama"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN"):
        return "huggingface"
    return "local"

async def _llm_chat_answer(user_message: str, session: dict, *, context_override: Optional[str] = None) -> str:
    provider = _default_provider()
    include_preview = (os.getenv("AI_ASSISTANT_INCLUDE_PREVIEW", "0") or "0").strip() == "1"
    context = context_override if isinstance(context_override, str) else _build_assistant_context(session, include_preview=include_preview)

    # Retry is only for transient failures (5xx/network). Avoid retrying 429 by default.
    max_retries = int(os.getenv("AI_ASSISTANT_MAX_RETRIES", "1"))
    base_delay_s = float(os.getenv("AI_ASSISTANT_RETRY_BASE_DELAY_S", "0.8"))

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            if provider == "openai":
                return await _openai_chat(user_message, context)
            if provider == "gemini":
                return await _gemini_chat(user_message, context)
            if provider == "ollama":
                return await _ollama_chat(user_message, context)
            if provider in ("huggingface", "hf"):
                return await _huggingface_chat(user_message, context)
            if provider == "local":
                return _local_chat_answer(user_message, session)
            raise ValueError(f"Unknown AI_ASSISTANT_PROVIDER={provider!r}. Use: gemini, openai, huggingface, ollama, or local.")
        except httpx.HTTPStatusError as e:
            last_err = e
            status = e.response.status_code
            if status == 429:
                # Do not auto-retry 429 by default (can amplify rate limiting).
                raise
            raise
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                await asyncio.sleep(min(base_delay_s * (2 ** attempt) + random.uniform(0, 0.25), 10.0))
                continue
            raise

    raise last_err or RuntimeError("Assistant failed.")

async def _index_session_data(session: dict):
    """Index session data into RAG vector store for better AI responses"""
    try:
        # Build context packet
        context_packet = build_context_packet(session)
        # Index into RAG system
        await retriever.index_session_data(context_packet)
        logging.info("[RAG] Session data indexed successfully")
    except Exception as e:
        logging.warning(f"[RAG] Failed to index session data: {e}")
        # Don't raise - indexing is optional enhancement

def _parse_combo_key(combo_key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(combo_key, str) or "|||" not in combo_key:
        return None, None
    parts = combo_key.split("|||", 1)
    if len(parts) != 2:
        return None, None
    sku = parts[0].strip() or None
    loc = parts[1].strip() or None
    return sku, loc

def _get_supply_plan_for_analysis(session: dict) -> Optional[pd.DataFrame]:
    full_df = session.get("supply_plan_full_df")
    if isinstance(full_df, pd.DataFrame) and not full_df.empty:
        return full_df
    export_df = session.get("supply_plan_df")
    if isinstance(export_df, pd.DataFrame) and not export_df.empty:
        return export_df
    return None

def _series_group_cols(df: pd.DataFrame) -> list[str]:
    if {"sku_id", "location"}.issubset(df.columns):
        return ["sku_id", "location"]
    if {"item", "store"}.issubset(df.columns):
        return ["item", "store"]
    return []

def build_planning_context(session: dict, combo_key: Optional[str] = None) -> dict[str, Any]:
    raw_df = session.get("df")
    forecast_df = session.get("forecast_df")
    plan_df = _get_supply_plan_for_analysis(session)

    context: dict[str, Any] = {
        "has_raw_data": isinstance(raw_df, pd.DataFrame) and not raw_df.empty,
        "has_forecast": isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty,
        "has_supply_plan": isinstance(plan_df, pd.DataFrame) and not plan_df.empty,
        "selected_combo_key": combo_key,
        "selected_sku": None,
        "selected_location": None,
        "raw_summary": None,
        "forecast_summary": None,
        "supply_plan_summary": None,
    }

    sku, loc = _parse_combo_key(combo_key)
    context["selected_sku"] = sku
    context["selected_location"] = loc

    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        context["raw_summary"] = _summarize_raw_data(raw_df)

    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        context["forecast_summary"] = _summarize_forecast(forecast_df)

    if isinstance(plan_df, pd.DataFrame) and not plan_df.empty:
        context["supply_plan_summary"] = _summarize_supply_plan(plan_df)
        context["supply_plan_columns"] = [str(c) for c in plan_df.columns]
        gcols = _series_group_cols(plan_df)
        context["series_group_cols"] = gcols
        if gcols:
            context["series_count"] = int(plan_df[gcols].drop_duplicates().shape[0])
        else:
            context["series_count"] = 1

    return context

def detect_risks(session: dict, context: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    plan_df = _get_supply_plan_for_analysis(session)
    if not isinstance(plan_df, pd.DataFrame) or plan_df.empty:
        return [{"type": "NO_SUPPLY_PLAN", "severity": "low", "message": "No supply plan found. Generate a supply plan first."}]

    work = plan_df.copy()
    gcols = _series_group_cols(work)

    # Normalize numeric columns if present
    for col in ["order_qty", "stockout_qty", "ending_on_hand", "inventory_position", "reorder_point", "target_level"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "period_start" in work.columns:
        work["period_start"] = pd.to_datetime(work["period_start"], errors="coerce")

    if context and context.get("selected_sku") and context.get("selected_location") and {"sku_id", "location"}.issubset(work.columns):
        work = work[
            (work["sku_id"].astype(str) == str(context["selected_sku"]))
            & (work["location"].astype(str) == str(context["selected_location"]))
        ]

    risks: list[dict[str, Any]] = []

    # Global risks
    if "stockout_qty" in work.columns:
        total_stockout = float(work["stockout_qty"].fillna(0.0).sum())
        if total_stockout > 0:
            risks.append({
                "type": "STOCKOUTS_PRESENT",
                "severity": "high",
                "message": f"Stockouts detected in the supply plan (total unmet demand: {_fmt_num(total_stockout)}).",
            })

    if gcols:
        grouped = work.groupby(gcols, as_index=False).agg({
            "stockout_qty": "sum" if "stockout_qty" in work.columns else "size",
            "order_qty": "sum" if "order_qty" in work.columns else "size",
        })
        if "stockout_qty" in grouped.columns:
            worst = grouped.sort_values("stockout_qty", ascending=False).head(5)
            for _, r in worst.iterrows():
                s_u = float(r.get("stockout_qty") or 0.0)
                if s_u <= 0:
                    continue
                series = {c: str(r[c]) for c in gcols}
                risks.append({
                    "type": "SERIES_STOCKOUT",
                    "severity": "high",
                    "series": series,
                    "message": f"Series has stockouts (unmet demand: {_fmt_num(s_u)}).",
                })

    # Capacity binding heuristic: order_qty close to cap_week*4 when both exist
    if "order_qty" in work.columns and "max_capacity_per_week" in work.columns:
        cap = pd.to_numeric(work["max_capacity_per_week"], errors="coerce")
        cap_month = cap * 4.0
        oq = work["order_qty"]
        hit = (cap_month.notna()) & (cap_month > 0) & (oq.notna()) & (oq >= 0.98 * cap_month)
        if bool(hit.any()):
            risks.append({
                "type": "CAPACITY_BINDING",
                "severity": "medium",
                "message": "Some orders are at/near monthly capacity (max_capacity_per_week * 4).",
            })

    # Ordering every month heuristic (could indicate review policy or low starting inventory)
    if "order_qty" in work.columns and "period_start" in work.columns and gcols:
        order_flags = work.assign(_ordered=(work["order_qty"].fillna(0.0) > 0))
        freq = order_flags.groupby(gcols, as_index=False)["_ordered"].mean()
        frequent = freq[freq["_ordered"] >= 0.95].head(10)
        for _, r in frequent.iterrows():
            series = {c: str(r[c]) for c in gcols}
            risks.append({
                "type": "FREQUENT_ORDERING",
                "severity": "low",
                "series": series,
                "message": "Orders occur nearly every month for this series (review cadence / low buffers).",
            })

    return risks

def generate_actions(session: dict, context: Optional[dict[str, Any]] = None, risks: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    plan_df = _get_supply_plan_for_analysis(session)
    if not isinstance(plan_df, pd.DataFrame) or plan_df.empty:
        return [{"type": "NO_ACTIONS", "message": "No supply plan found. Generate a supply plan first."}]

    context = context or build_planning_context(session)
    risks = risks or detect_risks(session, context)

    actions: list[dict[str, Any]] = []

    # If stockouts exist, suggest increasing starting on-hand or reducing lead time
    if any(r.get("type") in {"STOCKOUTS_PRESENT", "SERIES_STOCKOUT"} for r in risks):
        actions.append({
            "type": "INCREASE_ON_HAND",
            "priority": 1,
            "message": "Consider increasing starting on-hand for affected series (or expediting inbound) to prevent stockouts before receipts arrive.",
            "suggested_overrides": {"override_on_hand": "increase"},
        })
        actions.append({
            "type": "REDUCE_LEAD_TIME",
            "priority": 2,
            "message": "If possible, reduce lead time (or expedite) for affected SKUs; the model buckets lead time to months.",
            "suggested_overrides": {"override_lead_time_days": "decrease"},
        })

    if any(r.get("type") == "CAPACITY_BINDING" for r in risks):
        actions.append({
            "type": "ADDRESS_CAPACITY",
            "priority": 2,
            "message": "Some orders are constrained by capacity; consider increasing max capacity or splitting orders across suppliers/weeks.",
            "suggested_overrides": {"override_max_capacity_per_week": "increase"},
        })

    if not actions:
        actions.append({
            "type": "NO_MAJOR_RISKS",
            "priority": 3,
            "message": "No major risks detected from the current supply plan. Review reorder point vs target level and validate inputs.",
        })

    return actions

def _local_chat_answer(user_message: str, session: dict) -> str:
    q = (user_message or "").strip()
    ql = q.lower()

    # Term definitions (supply plan)
    for term, meaning in _SUPPLY_PLAN_GLOSSARY.items():
        if term in ql:
            return f"`{term}`: {meaning}"

    raw_df = session.get("df")
    forecast_df = session.get("forecast_df")
    supply_plan_df = session.get("supply_plan_df")

    has_raw = isinstance(raw_df, pd.DataFrame) and not raw_df.empty
    has_forecast = isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty
    has_plan = isinstance(supply_plan_df, pd.DataFrame) and not supply_plan_df.empty

    if any(k in ql for k in ["help", "what can you do", "examples", "how to use"]):
        return (
            "Ask me about:\n"
            "- Raw data: date range, totals, top items/stores, missing values\n"
            "- Forecast: totals, top month, top items/stores\n"
            "- Supply plan: total orders, stockouts, top items/stores by order\n\n"
            "Examples: “raw data summary”, “top items by sales”, “total forecast”, “stockouts in supply plan”."
        )

    if not (has_raw or has_forecast or has_plan):
        return "I don't have raw data/forecast/supply plan in this session yet. Upload data, run a forecast, and generate a supply plan first."

    wants_supply = any(k in ql for k in ["supply", "order", "reorder", "stockout", "moq", "lead time", "inventory", "target_level", "reorder_point"])
    wants_forecast = "forecast" in ql or "prediction" in ql
    wants_raw = any(k in ql for k in ["raw", "uploaded", "dataset", "sales", "actual"]) and not wants_supply

    if wants_supply and has_plan:
        return _summarize_supply_plan(supply_plan_df)
    if wants_forecast and has_forecast:
        return _summarize_forecast(forecast_df)
    if wants_raw and has_raw:
        return _summarize_raw_data(raw_df)

    # Default: show a compact combined summary.
    blocks: list[str] = []
    if has_raw:
        blocks.append(_summarize_raw_data(raw_df))
    if has_forecast:
        blocks.append(_summarize_forecast(forecast_df))
    if has_plan:
        blocks.append(_summarize_supply_plan(supply_plan_df))

    return " ".join(blocks[:3])

# Helper to update progress
def set_forecast_progress(session_id, progress, status=None, done=False, error=None):
    if session_id not in data_store:
        data_store[session_id] = {}
    existing = data_store[session_id].get("forecast_progress") or {}
    logging.debug(f"[LOG] set_forecast_progress: session_id={session_id}, progress={progress}, status={status}, done={done}, error={error}")
    data_store[session_id]["forecast_progress"] = {
        "progress": progress,
        "status": status or "",
        "done": done,
        "error": error or "",
        # Preserve cancellation state unless explicitly set elsewhere.
        "cancelled": bool(existing.get("cancelled", False)),
    }

class ForecastCancelled(Exception):
    pass

def _forecast_cancel_requested(session_id: str) -> bool:
    return bool(data_store.get(session_id, {}).get("forecast_cancel_requested", False))

def _set_forecast_cancel_requested(session_id: str, requested: bool) -> None:
    s = data_store.setdefault(session_id, {})
    s["forecast_cancel_requested"] = bool(requested)

def _set_forecast_cancelled(session_id: str, status: str = "Forecast cancelled.") -> None:
    if session_id not in data_store:
        data_store[session_id] = {}
    prev = data_store[session_id].get("forecast_progress") or {}
    data_store[session_id]["forecast_progress"] = {
        "progress": float(prev.get("progress", 0.0) or 0.0),
        "status": status,
        "done": True,
        "error": "",
        "cancelled": True,
    }

# Endpoint for polling forecast progress
@app.get("/forecast_status")
async def forecast_status(request: Request):
    session_id = _session_id_from_request(request)
    if session_id in data_store:
        logging.error("===== STATUS ROUTE DF =====")
        logging.error(data_store[session_id]["df"].columns.tolist())
        logging.error(data_store[session_id]["df"].shape)
    prog = data_store.get(session_id, {}).get("forecast_progress", None)
    forecast_ready = (
        session_id in data_store and
        "forecast_df" in data_store[session_id] and
        isinstance(data_store[session_id]["forecast_df"], pd.DataFrame) and
        not data_store[session_id]["forecast_df"].empty
    )
    # print(f"[LOG] /forecast_status: progress={prog}, forecast_ready={forecast_ready}")
    if not prog:
        logging.debug("[LOG] /forecast_status: No progress found, returning not started.")
        return {"progress": 0, "status": "Not started", "done": False}
    cancelled = bool(prog.get("cancelled", False))
    # Only set done: true if forecast is actually ready (unless cancelled)
    result = dict(prog)
    result["cancel_requested"] = _forecast_cancel_requested(session_id)
    result["cancelled"] = cancelled
    if cancelled:
        result["done"] = True
    else:
        result["done"] = prog.get("done", False) and forecast_ready
    logging.debug(f"[LOG] /forecast_status: Returning {result}")
    return result

@app.post("/forecast_stop")
async def forecast_stop(request: Request):
    session_id = _session_id_from_request(request)
    _set_forecast_cancel_requested(session_id, True)
    prog = data_store.get(session_id, {}).get("forecast_progress") or {}
    if prog and not prog.get("done", False):
        set_forecast_progress(session_id, prog.get("progress", 0.0) or 0.0, "Cancelling...")
    return {"ok": True}


@app.post("/request_demo")
async def request_demo(request: Request, background_tasks: BackgroundTasks):
    """
    Save a request-demo lead. No LLM; best-effort persistence to history DB.
    """
    try:
        payload = {}
        ctype = (request.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            payload = await request.json()
        else:
            form = await request.form()
            payload = dict(form)
    except Exception:
        payload = {}

    # Honeypot: bots tend to fill hidden fields.
    if str(payload.get("hp") or "").strip():
        return JSONResponse({"ok": True})

    name = str(payload.get("name") or "").strip()
    email = str(payload.get("email") or "").strip()
    company = str(payload.get("company") or "").strip()
    phone = str(payload.get("phone") or "").strip()
    role = str(payload.get("role") or "").strip()
    message = str(payload.get("message") or "").strip()
    source_path = str(payload.get("source_path") or request.url.path or "").strip()
    user_email = _get_user_email(request)

    if not name or "@" not in email:
        return JSONResponse({"ok": False, "error": "Name and a valid email are required."}, status_code=400)

    try:
        import history_store

        history_store.save_demo_request(
            name=name,
            email=email,
            company=company or None,
            phone=phone or None,
            role=role or None,
            message=message or None,
            source_path=source_path or None,
            user_email=user_email or None,
        )
    except Exception as e:
        logging.warning(f"[DEMO] Failed to save demo request: {e}")
        # Still return ok to avoid blocking UX.

    # Best-effort email notification (does not affect response).
    if (
        (os.getenv("PITENSOR_SMTP_HOST") or "").strip()
        or (os.getenv("PITENSOR_RESEND_API_KEY") or "").strip()
        or (os.getenv("PITENSOR_SENDGRID_API_KEY") or "").strip()
    ):
        def _email_task() -> None:
            try:
                _send_demo_request_notification(
                    name=name,
                    email=email,
                    company=company or None,
                    phone=phone or None,
                    role=role or None,
                    message=message or None,
                    source_path=source_path or None,
                    user_email=user_email or None,
                )
            except Exception as e:
                provider = (os.getenv("PITENSOR_EMAIL_PROVIDER") or "").strip()
                smtp_host = (os.getenv("PITENSOR_SMTP_HOST") or "").strip()
                smtp_port = (os.getenv("PITENSOR_SMTP_PORT") or "587").strip()
                use_ssl = (os.getenv("PITENSOR_SMTP_USE_SSL") or "0").strip()
                use_tls = (os.getenv("PITENSOR_SMTP_USE_TLS") or "1").strip()
                timeout_s = (os.getenv("PITENSOR_EMAIL_TIMEOUT_S") or os.getenv("PITENSOR_SMTP_TIMEOUT_S") or "12").strip()
                extra = ""
                if isinstance(e, httpx.HTTPStatusError) and getattr(e, "response", None) is not None:
                    try:
                        extra = f" status={e.response.status_code} body={e.response.text[:500]}"
                    except Exception:
                        extra = f" status={getattr(e.response, 'status_code', '')}"
                logging.warning(
                    f"[DEMO] Failed to send demo request email: {e} (provider={provider or 'auto'} host={smtp_host} port={smtp_port} ssl={use_ssl} tls={use_tls} timeout_s={timeout_s}){extra}"
                )

        background_tasks.add_task(_email_task)
    return JSONResponse({"ok": True})


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/signin", response_class=HTMLResponse)
async def signin_page(request: Request, next: Optional[str] = None, error: Optional[str] = None):
    return templates.TemplateResponse(
        "signin.html",
        {
            "request": request,
            "next": next,
            "error": error,
            "stage": "email",
            "email": "",
            "otp_id": "",
            "info": "",
            "auth_mode": _auth_mode(),
        },
    )

@app.post("/signin")
async def signin_post(
    request: Request,
    background_tasks: BackgroundTasks,
    email: str = Form(...),
    next: Optional[str] = Form(default=None),
):
    email = (email or "").strip()
    if "@" not in email or "." not in email.split("@")[-1]:
        return templates.TemplateResponse(
            "signin.html",
            {
                "request": request,
                "next": next,
                "error": "Please enter a valid email address.",
                "stage": "email",
                "email": email,
                "otp_id": "",
                "info": "",
                "auth_mode": _auth_mode(),
            },
            status_code=400,
        )

    if _auth_mode() == "otp":
        try:
            last_dt = None
            try:
                import history_store

                # Throttle: avoid spamming OTPs.
                last_created = history_store.get_latest_auth_otp_created_at(email=email)
                if last_created:
                    last_dt = datetime.fromisoformat(str(last_created))
            except Exception:
                last_dt = None

            if last_dt is not None:
                try:
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    if (datetime.now(timezone.utc) - last_dt).total_seconds() < 30:
                        return templates.TemplateResponse(
                            "signin.html",
                            {
                                "request": request,
                                "next": next,
                                "error": "Please wait a moment before requesting another code.",
                                "stage": "email",
                                "email": email,
                                "otp_id": "",
                                "info": "",
                                "auth_mode": _auth_mode(),
                            },
                            status_code=429,
                        )
                except Exception:
                    pass

            otp_id = _b64url_encode(os.urandom(18))
            otp_code = f"{int.from_bytes(os.urandom(3), 'big') % 1000000:06d}"
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).replace(microsecond=0).isoformat()
            code_hash = _otp_hash(otp_id, otp_code)
            try:
                import history_store

                history_store.create_auth_otp(
                    otp_id=otp_id,
                    email=email,
                    code_hash=code_hash,
                    expires_at=expires_at,
                    ip=_client_ip(request),
                    user_agent=(request.headers.get("user-agent") or "")[:300],
                )
            except Exception:
                _otp_mem_put(otp_id=otp_id, email=email.lower(), code_hash=code_hash, expires_at=expires_at)

            try:
                _send_text_email(
                    to_addr=email,
                    subject="Your PiTensor sign-in code",
                    body=(
                        "Your PiTensor sign-in code is:\n\n"
                        f"{otp_code}\n\n"
                        "It expires in 10 minutes.\n\n"
                        "If you did not request this, you can ignore this email."
                    ),
                )
            except Exception as e:
                extra = ""
                if isinstance(e, httpx.HTTPStatusError) and getattr(e, "response", None) is not None:
                    try:
                        extra = f" status={e.response.status_code} body={e.response.text[:500]}"
                    except Exception:
                        extra = f" status={getattr(e.response, 'status_code', '')}"
                logging.warning(f"[AUTH] Failed to send OTP email: {e}{extra}")
                return templates.TemplateResponse(
                    "signin.html",
                    {
                        "request": request,
                        "next": next,
                        "error": f"Could not send sign-in code: {_user_friendly_email_send_error(e)}",
                        "stage": "email",
                        "email": email,
                        "otp_id": "",
                        "info": "",
                        "auth_mode": _auth_mode(),
                    },
                    status_code=500,
                )

            return templates.TemplateResponse(
                "signin.html",
                {
                    "request": request,
                    "next": next,
                    "error": None,
                    "stage": "otp",
                    "email": email,
                    "otp_id": otp_id,
                    "info": "We sent a 6-digit code to your email. Enter it below.",
                    "auth_mode": _auth_mode(),
                },
            )
        except Exception as e:
            return templates.TemplateResponse(
                "signin.html",
                {
                    "request": request,
                    "next": next,
                    "error": f"Could not send sign-in code: {e}",
                    "stage": "email",
                    "email": email,
                    "otp_id": "",
                    "info": "",
                    "auth_mode": _auth_mode(),
                },
                status_code=500,
            )

    dest = _safe_next_path(next)
    if dest == "/" and not (next or "").strip():
        dest = "/dashboard"
    response = RedirectResponse(dest, status_code=303)
    response.set_cookie(
        _auth_cookie_name,
        _auth_cookie_value(email),
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
        max_age=_auth_cookie_max_age_s or None,
    )
    return response


@app.post("/signin/verify")
async def signin_verify(
    request: Request,
    otp_id: str = Form(...),
    email: str = Form(...),
    code: str = Form(...),
    next: Optional[str] = Form(default=None),
):
    if _auth_mode() != "otp":
        return RedirectResponse("/signin", status_code=303)

    otp_id = (otp_id or "").strip()
    email = (email or "").strip().lower()
    code = "".join([c for c in (code or "") if c.isdigit()]).strip()
    if not otp_id or "@" not in email or len(code) != 6:
        return templates.TemplateResponse(
            "signin.html",
            {
                "request": request,
                "next": next,
                "error": "Please enter the 6-digit code.",
                "stage": "otp",
                "email": email,
                "otp_id": otp_id,
                "info": "",
                "auth_mode": _auth_mode(),
            },
            status_code=400,
        )

    try:
        rec = {}
        use_mem = False
        try:
            import history_store

            rec = history_store.get_auth_otp(otp_id=otp_id) or {}
        except Exception:
            rec = _otp_mem_get(otp_id=otp_id) or {}
            use_mem = True

        if not rec:
            raise ValueError("Code expired or invalid.")
        if str(rec.get("email") or "").strip().lower() != email:
            raise ValueError("Code expired or invalid.")
        if rec.get("consumed_at"):
            raise ValueError("Code already used.")
        attempts = int(rec.get("attempts") or 0)
        if attempts >= 5:
            raise ValueError("Too many attempts. Please request a new code.")
        expires_at = str(rec.get("expires_at") or "").strip()
        if expires_at:
            try:
                exp_dt = datetime.fromisoformat(expires_at)
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) > exp_dt:
                    raise ValueError("Code expired. Please request a new code.")
            except ValueError:
                raise
            except Exception:
                pass

        expected = str(rec.get("code_hash") or "")
        actual = _otp_hash(otp_id, code)
        if not expected or not hmac.compare_digest(expected, actual):
            try:
                if use_mem:
                    _otp_mem_inc_attempts(otp_id=otp_id)
                else:
                    import history_store

                    history_store.increment_auth_otp_attempts(otp_id=otp_id)
            except Exception:
                _otp_mem_inc_attempts(otp_id=otp_id)
            raise ValueError("Incorrect code.")

        try:
            if use_mem:
                _otp_mem_consume(otp_id=otp_id)
            else:
                import history_store

                history_store.consume_auth_otp(otp_id=otp_id)
        except Exception:
            _otp_mem_consume(otp_id=otp_id)

    except Exception as e:
        return templates.TemplateResponse(
            "signin.html",
            {
                "request": request,
                "next": next,
                "error": str(e) or "Invalid code.",
                "stage": "otp",
                "email": email,
                "otp_id": otp_id,
                "info": "",
                "auth_mode": _auth_mode(),
            },
            status_code=400,
        )

    dest = _safe_next_path(next)
    if dest == "/" and not (next or "").strip():
        dest = "/dashboard"
    response = RedirectResponse(dest, status_code=303)
    response.set_cookie(
        _auth_cookie_name,
        _auth_cookie_value(email),
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
        max_age=_auth_cookie_max_age_s or None,
    )
    return response

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request, next: Optional[str] = None, error: Optional[str] = None):
    # For now, route sign-up through the same flow as sign-in.
    dest = "/signin"
    if (next or "").strip():
        dest = f"/signin?next={quote(str(next))}"
    return RedirectResponse(dest, status_code=303)

@app.post("/signup")
async def signup_post(request: Request):
    return RedirectResponse("/signin", status_code=303)

@app.get("/signout")
async def signout(request: Request):
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(_auth_cookie_name)
    return response

@app.get("/auth/google")
async def auth_google(request: Request, next: Optional[str] = None):
    # OAuth not configured in this repo yet; show sign-in UI with message.
    return RedirectResponse(f"/signin?error={quote('Google sign-in is not configured yet.')}&next={quote(next or '')}", status_code=303)

@app.get("/auth/github")
async def auth_github(request: Request, next: Optional[str] = None):
    return RedirectResponse(f"/signin?error={quote('GitHub sign-in is not configured yet.')}&next={quote(next or '')}", status_code=303)

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV")
    contents = await file.read()
    
    # Try different delimiters to handle various CSV formats
    possible_delimiters = [',', '\t', ';', '|']
    df = None
    
    for delimiter in possible_delimiters:
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=delimiter)
            # Check if we have the required columns
            if {'date', 'item', 'store', 'sales'}.issubset(set(df.columns)):
                print(f"Successfully parsed CSV with delimiter: '{delimiter}'")
                break
        except:
            continue
    
    if df is None or not {'date', 'item', 'store', 'sales'}.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail="Could not parse CSV file or missing required columns: date, item, store, sales")
    
    # Debug logging
    logging.debug(f"Uploaded file: {file.filename}")
    logging.debug(f"DataFrame shape: {df.shape}")
    logging.debug(f"Columns found: {df.columns.tolist()}")
    logging.debug("First few rows:\n%s", df.head(3))
    
    # Additional validation
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format in 'date' column: {str(e)}")
    
    session_id = _session_id_from_request(request)
    stored_df = df.copy()
    user_email = _get_user_email(request)
    dataset_id = None
    history_last_error = None
    if user_email:
        try:
            import history_store
            dataset_id = history_store.save_dataset(user_email, file.filename, stored_df)
        except Exception as e:
            logging.warning(f"[HISTORY] Failed to save dataset for {user_email}: {e}")
            history_last_error = f"Failed to save dataset history: {e}"

    data_store[session_id] = {
        "df": stored_df,
        "raw_df": stored_df.copy(),
        "uploaded_filename": file.filename,
        "dataset_id": dataset_id,
        "history_last_error": history_last_error,
    }
    return RedirectResponse("/forecast", status_code=303)

@app.get("/forecast", response_class=HTMLResponse)
async def forecast_page(request: Request):
    session_id = _session_id_from_request(request)
    if session_id not in data_store or "df" not in data_store[session_id]:
        return RedirectResponse("/?error=no_data", status_code=303)

    df = data_store[session_id]["df"]
    logging.error("===== BEFORE FORECAST DF =====")
    logging.error(df.columns.tolist())
    logging.error(df.shape)

    max_date = df['date'].max()
    last_month = max_date.strftime('%Y-%m')

    # Calculate default forecast start month (3 months before last data month)
    from dateutil.relativedelta import relativedelta
    default_start_month = (max_date - relativedelta(months=3)).strftime('%Y-%m')

    # Get numeric/boolean columns for OOS selection
    exclude_cols = {'date', 'sales', 'item', 'store'}
    numeric_columns = []
    for col in df.columns:
        if col not in exclude_cols:
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                numeric_columns.append(col)

    return templates.TemplateResponse("forecast.html", {
        "request": request,
        "data_shape": df.shape,
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_columns,
        "date_range": f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        "last_month": last_month,
        "default_start_month": default_start_month
    })

@app.get("/status")
async def get_status(request: Request):
    session_id = _session_id_from_request(request)
    status = {
        "session_exists": session_id in data_store,
        "data_keys": list(data_store.get(session_id, {}).keys()) if session_id in data_store else [],
        "data_shape": data_store[session_id]["df"].shape if session_id in data_store and "df" in data_store[session_id] else None,
        "horizon": data_store[session_id].get("horizon"),  # Keep for backward compatibility
        "start_month": data_store[session_id].get("start_month"),
        "months": data_store[session_id].get("months"),
        "forecast_ready": "forecast_df" in data_store.get(session_id, {}),
        "forecast_shape": data_store[session_id]["forecast_df"].shape if session_id in data_store and "forecast_df" in data_store[session_id] else None
    }
    return status


@app.get("/session/settings")
async def get_session_settings(session_id: str = "default"):
    """Return session-level settings (including seasonal feature flags)."""
    session = data_store.get(session_id, {})
    # If the session has no saved settings, return an empty seasonal_flags
    # so the UI checkboxes remain unchecked by default until the user saves.
    settings = session.get("session_settings") or {"seasonal_flags": {}}
    return settings


@app.post("/session/settings")
async def set_session_settings(payload: Dict = Body(default={}), session_id: str = "default"):
    """Persist session settings and mirror seasonal flags to the running config.
    Expected payload: { "seasonal_flags": { ... } }
    """
    session = data_store.setdefault(session_id, {})
    s = session.setdefault("session_settings", {})
    sf = payload.get("seasonal_flags")
    if isinstance(sf, dict):
        s["seasonal_flags"] = sf
        try:
            # Mirror into global config for runtime modules that read config.SEASONAL_FLAGS
            import config as _config
            _config.SEASONAL_FLAGS = sf
        except Exception:
            pass
    return {"session_settings": s}

@app.get("/loading", response_class=HTMLResponse)
async def loading_page(request: Request):
    return templates.TemplateResponse("loading.html", {"request": request})

from typing import List

@app.post("/forecast")
async def run_forecast(
    request: Request,
    start_month: str = Form(...),
    months: int = Form(...),
    grain: List[str] = Form(None),
    extra_features: List[str] = Form(None),
    enable_oos_imputation: Optional[str] = Form(None),
    oos_column: Optional[str] = Form(None)
):
    session_id = _session_id_from_request(request)
    user_email = _get_user_email(request)
    print(f"=== FORECAST FORM SUBMITTED ===")
    logging.debug(f"Start Month: {start_month}")
    logging.debug(f"Months: {months}")
    logging.debug(f"Session data exists: {session_id in data_store}")
    if session_id in data_store:
        logging.debug(f"Session data keys: {list(data_store[session_id].keys())}")
    else:
        logging.error("ERROR: No session data found!")
        return RedirectResponse("/?error=no_data", status_code=303)

    # Store OOS settings
    oos_enabled = enable_oos_imputation == "on"
    oos_col = oos_column if (oos_enabled and oos_column) else None

    data_store[session_id]["start_month"] = start_month
    data_store[session_id]["months"] = months
    data_store[session_id]["grain"] = grain if grain else []
    data_store[session_id]["extra_features"] = extra_features if extra_features else []
    data_store[session_id]["oos_enabled"] = oos_enabled
    data_store[session_id]["oos_column"] = oos_col
    logging.debug(f"Start month stored in session: {data_store[session_id].get('start_month')}")
    logging.debug(f"Months stored in session: {data_store[session_id].get('months')}")
    logging.debug(f"Forecast grain columns stored in session: {data_store[session_id].get('grain')}")
    logging.debug(f"Extra features for training: {data_store[session_id].get('extra_features')}")
    logging.debug(f"OOS imputation enabled: {oos_enabled}, column: {oos_col}")

    # Reset cancel state for a new run.
    _set_forecast_cancel_requested(session_id, False)
    try:
        if "forecast_progress" in data_store.get(session_id, {}):
            data_store[session_id]["forecast_progress"]["cancelled"] = False
    except Exception:
        pass

    # Start forecast in a background thread and return immediately
    def forecast_task():
        logging.debug(f"[LOG] Forecast thread started for session_id={session_id}")
        try:
            if _forecast_cancel_requested(session_id):
                _set_forecast_cancelled(session_id, "Forecast cancelled.")
                return
            set_forecast_progress(session_id, 0.05, "Preparing data...")
            raw_df = data_store[session_id].get("raw_df")
            if raw_df is None:
                raw_df = data_store[session_id]["df"]
            df = raw_df.copy()
            data_store[session_id]["df"] = df
            logging.error("===== UPLOAD STORED DF =====")
            logging.error(data_store[session_id]["df"].columns.tolist())
            logging.error(data_store[session_id]["df"].shape)

            extra_features = data_store[session_id].get("extra_features", [])
            grain = data_store[session_id].get("grain", [])

            # 🔥 FIX: Auto-detect grain if empty
            if not grain:
                grain = [col for col in ["item", "store"] if col in df.columns]
                logging.debug(f"[FIX] Grain auto-detected as: {grain}")

            logging.debug(f"[DEBUG] DF retrieved from data_store. Shape: {df.shape if df is not None else None}")
            logging.debug(f"[DEBUG] DF columns: {df.columns.tolist() if df is not None else None}")
            logging.debug(f"[DEBUG] Grain parameter: {grain}")
            logging.debug(f"[DEBUG] Extra features parameter: {extra_features}")
            if df is None or df.empty:
                set_forecast_progress(session_id, 1.0, "No data available", done=True, error="No data available for forecasting")
                logging.debug(f"[LOG] Forecast thread: No data available, exiting.")
                return

            if _forecast_cancel_requested(session_id):
                _set_forecast_cancelled(session_id, "Forecast cancelled.")
                return
            start_date = pd.to_datetime(start_month + "-01")
            from run_forecast2 import forecast_all_combined_prob, impute_oos_sales

            # Conditional OOS imputation
            oos_enabled = data_store[session_id].get("oos_enabled", False)
            oos_col = data_store[session_id].get("oos_column")
            if oos_enabled and oos_col and oos_col in df.columns:
                set_forecast_progress(session_id, 0.15, "Imputing out-of-stock sales...")
                # Store original sales data before imputation
                data_store[session_id]["original_sales_df"] = df.copy()
                df = impute_oos_sales(df, grain=grain, oos_col=oos_col)
                data_store[session_id]["oos_imputed"] = True
                logging.debug(f"[LOG] OOS imputation performed using column: {oos_col}")
            else:
                data_store[session_id]["oos_imputed"] = False
                data_store[session_id]["original_sales_df"] = None
                logging.debug(f"[LOG] OOS imputation skipped (enabled={oos_enabled}, column={oos_col})")

            if _forecast_cancel_requested(session_id):
                _set_forecast_cancelled(session_id, "Forecast cancelled.")
                return

            set_forecast_progress(session_id, 0.35, "Running forecast model...")
            df = data_store[session_id]["df"].copy()
            extra_features = data_store[session_id].get("extra_features", [])
            grain = data_store[session_id].get("grain", [])
            print("gg grain before calling combined prob", grain)

            def _progress_cb(p, msg=None):
                if _forecast_cancel_requested(session_id):
                    raise ForecastCancelled()
                set_forecast_progress(session_id, 0.35 + 0.6 * p, msg or "Forecasting...")

            forecast_df, feature_importance, driver_artifacts = forecast_all_combined_prob(
                df, start_date=start_date, months=months, grain=grain, extra_features=extra_features,
                progress_callback=_progress_cb
            )
            logging.debug(f"[LOG] Forecast thread: forecast_df shape={getattr(forecast_df, 'shape', None)}")
            data_store[session_id]["forecast_df"] = forecast_df
            data_store[session_id]["feature_importance"] = feature_importance
            data_store[session_id]["driver_artifacts"] = driver_artifacts or {}

            # Persist run history (best-effort; does not affect the forecast flow).
            if user_email:
                try:
                    import history_store
                    params = {
                        "start_month": start_month,
                        "months": months,
                        "grain": data_store.get(session_id, {}).get("grain", []),
                        "extra_features": data_store.get(session_id, {}).get("extra_features", []),
                        "oos_enabled": bool(data_store.get(session_id, {}).get("oos_enabled", False)),
                        "oos_column": data_store.get(session_id, {}).get("oos_column"),
                        "uploaded_filename": data_store.get(session_id, {}).get("uploaded_filename"),
                    }
                    ds_id = data_store.get(session_id, {}).get("dataset_id")
                    if not ds_id and isinstance(data_store.get(session_id, {}).get("raw_df"), pd.DataFrame):
                        ds_id = history_store.save_dataset(
                            user_email,
                            str(data_store.get(session_id, {}).get("uploaded_filename") or "uploaded.csv"),
                            data_store[session_id]["raw_df"],
                        )
                        data_store[session_id]["dataset_id"] = ds_id
                    run_id = history_store.save_forecast_run(
                        user_email,
                        ds_id,
                        params=params,
                        forecast_df=forecast_df,
                        feature_importance=feature_importance,
                        driver_artifacts=(driver_artifacts or {}),
                    )
                    data_store[session_id]["forecast_run_id"] = run_id
                    data_store[session_id]["history_last_error"] = None
                except Exception as e:
                    logging.warning(f"[HISTORY] Failed to save forecast run for {user_email}: {e}")
                    try:
                        data_store[session_id]["history_last_error"] = f"Failed to save forecast run history: {e}"
                    except Exception:
                        pass

            # Index data for RAG system
            try:
                asyncio.run(_index_session_data(data_store[session_id]))
            except Exception as e:
                logging.warning(f"[LOG] Failed to index data for RAG: {e}")

            set_forecast_progress(session_id, 1.0, "Forecast complete!", done=True)
            logging.debug("[LOG] Forecast generated and stored in session.")
        except ForecastCancelled:
            _set_forecast_cancelled(session_id, "Forecast cancelled.")
        except Exception as e:
            logging.exception(f"[LOG] ERROR during forecast generation: {e}")
            set_forecast_progress(session_id, 1.0, "Error during forecast", done=True, error=str(e))
        logging.debug(f"[LOG] Forecast thread ended for session_id={session_id}")

    set_forecast_progress(session_id, 0.01, "Starting forecast...")
    logging.debug(f"[LOG] Starting forecast thread for session_id={session_id}")
    thread = threading.Thread(target=forecast_task)
    thread.start()
    logging.debug(f"[LOG] Forecast thread launched for session_id={session_id}")
    return JSONResponse({"started": True})

@app.post("/generate_forecast")
@app.post("/generate_forecast")
async def generate_forecast(request: Request):
    session_id = _session_id_from_request(request)
    print(f"=== gg GENERATE_FORECAST ENDPOINT CALLED ===")
    # Debug logging
    logging.debug("=== GENERATE_FORECAST STARTED ===")
    logging.debug(f"Session ID: {session_id}")
    logging.debug(f"Data store keys: {list(data_store.keys()) if data_store else 'Empty'}")
    if session_id in data_store:
        print(f"Session data keys: {list(data_store[session_id].keys())}")

    if session_id not in data_store or "start_month" not in data_store[session_id] or "months" not in data_store[session_id]:
        error_msg = "No data or forecast parameters available"
        print(f"ERROR: {error_msg}")
        return {"status": "error", "message": error_msg}

    try:
        df = data_store[session_id]["df"]
        logging.error("===== BEFORE FORECAST DF =====")
        logging.error(df.columns.tolist())
        logging.error(df.shape)

        # Additional validation
        if df is None or df.empty:
            raise ValueError("No data available for forecasting")

        # print(f"DataFrame shape: {df.shape}")
        # print(f"DataFrame columns: {df.columns.tolist()}")
        # print(f"DataFrame dtypes: {df.dtypes.to_dict()}")
        # print(f"First 3 rows:")
        # print(df.head(3))

        start_month_str = data_store[session_id]["start_month"]
        months = data_store[session_id]["months"]
        start_date = pd.to_datetime(start_month_str + "-01")  # First day of the month
        print("Starting forecast generation with start_date: {start_date}, months: {months}")
        grain = data_store[session_id].get("grain", ["item", "store"])
        print(f"gg Forecasting at grain: {grain}")

        from run_forecast2 import forecast_all_combined, impute_oos_sales

        # Conditional OOS imputation
        oos_enabled = data_store[session_id].get("oos_enabled", False)
        oos_col = data_store[session_id].get("oos_column")
        if oos_enabled and oos_col and oos_col in df.columns:
            print(f"Performing OOS imputation using column: {oos_col}")
            # Store original sales data before imputation
            data_store[session_id]["original_sales_df"] = df.copy()
            df = impute_oos_sales(df, grain=grain, oos_col=oos_col)
            data_store[session_id]["oos_imputed"] = True
        else:
            data_store[session_id]["oos_imputed"] = False
            data_store[session_id]["original_sales_df"] = None
            print(f"OOS imputation skipped (enabled={oos_enabled}, column={oos_col})")

        # print("gg 3 forecast df unique country", df['Country'].unique())
        if not grain or len(grain) == 0:
            grain = [col for col in ["item", "store"] if col in df.columns]
        uvicorn_logger = logging.getLogger("uvicorn.error")
        uvicorn_logger.info("gg grain before all combined called %s", grain)
        forecast_df, feature_importance = forecast_all_combined(df, start_date=start_date, months=months, grain=grain)
        
        # print("df head gg", df.head(3))
        print(f"Forecast generation completed successfully!")
        print(f"Forecast DF shape: {forecast_df.shape}")
        print(f"Feature importance keys: {list(feature_importance.keys())[:3] if feature_importance else 'None'}")

        data_store[session_id]["forecast_df"] = forecast_df
        data_store[session_id]["feature_importance"] = feature_importance

        # Index data for RAG system
        try:
            await _index_session_data(data_store[session_id])
        except Exception as e:
            logging.warning(f"Failed to index data for RAG: {e}")

        print("=== GENERATE_FORECAST COMPLETED SUCCESSFULLY ===")
        print(f"Returning success response")
        return {"status": "success", "message": "Forecast generated successfully"}

    except ValueError as e:
        # Handle validation errors with user-friendly messages
        error_msg = str(e)
        print(f"Validation error: {error_msg}")
        return {"status": "error", "message": error_msg}
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"An unexpected error occurred: {str(e)}"
        print(f"Unexpected error generating forecast: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": error_msg}

@app.get("/results", response_class=HTMLResponse)
async def get_results(request: Request):
    session_id = _session_id_from_request(request)
    logging.debug(f"[LOG] === RESULTS PAGE REQUESTED ===")
    logging.debug(f"[LOG] Session data keys: {list(data_store.get(session_id, {}).keys())}")
    if "forecast_df" not in data_store.get(session_id, {}):
        logging.error("[LOG] ERROR: No forecast_df in session")
        raise HTTPException(status_code=400, detail="No forecast available")
    forecast_df = data_store[session_id]["forecast_df"]
    feature_importance = data_store[session_id].get("feature_importance", {})
    grain = data_store[session_id].get("grain", ["item", "store"])
    raw_df = data_store.get(session_id, {}).get("df")
    logging.debug(f"[LOG] Using grain columns for filtering: {grain}")
    logging.debug(f"[LOG] Forecast DF shape: {getattr(forecast_df, 'shape', None)}")
    logging.debug(f"[LOG] Forecast DF columns: {getattr(forecast_df, 'columns', None)}")
    logging.debug(f"[LOG] Feature importance available: {bool(feature_importance)}")
    # Validate data
    if forecast_df.empty:
        logging.error("[LOG] ERROR: Forecast data is empty")
        raise HTTPException(status_code=400, detail="Forecast data is empty")
    

    # Build dynamic filters for each grain column
    all_grain_values = {}
    selected_grain_values = {}
    filter_mask = pd.Series([True] * len(forecast_df))
    # print(forecast_df.head(3))
    for col in grain:
        if col in forecast_df.columns:
            # Remove actual nan values before converting to string
            # print("gg forecast df unique country", forecast_df['Country'].unique())
            non_nan_vals = forecast_df[col][~forecast_df[col].isna()]
            all_vals = sorted(non_nan_vals.astype(str).unique())
            forecast_df[col] = forecast_df[col].astype(str)
            selected_vals = request.query_params.getlist(col) or all_vals
            all_grain_values[col] = all_vals
            selected_grain_values[col] = selected_vals
            filter_mask &= forecast_df[col].isin(selected_vals)
    # print(f"all_grain_values: {all_grain_values}")
    filtered_df = forecast_df[filter_mask].copy()
    if filtered_df.empty:
        print("ERROR: No data matches the selected filters")
        raise HTTPException(status_code=400, detail="No data matches the selected filters")
    
    print(f"Filtered data shape: {filtered_df.shape}")
    print("=== RESULTS PAGE RENDERING SUCCESSFUL ===")
    
    # Aggregate
    groupby_cols = [col for col in ["date"] + grain if col in filtered_df.columns]
    # Use forecast_p60 as the main forecast column if present, else fallback to forecast/other quantile
    # Find all quantile columns and keep original forecast if present
    quantile_cols = [col for col in filtered_df.columns if col.startswith("forecast_p")]
    agg_dict = {"actual": "sum"}
    if "forecast" in filtered_df.columns:
        agg_dict["forecast"] = "sum"
    for q in quantile_cols:
        agg_dict[q] = "sum"
    daily_agg = (
        filtered_df
        .groupby(groupby_cols, as_index=False)
        .agg(agg_dict)
    )
    monthly_agg = (
        daily_agg
        .set_index("date")
        .resample("MS")  # Month Start instead of Month End for clearer x-axis labels
        .sum()
        .reset_index()
    )
    # KPIs
    actual_total = daily_agg["actual"].sum()
    # Use forecast if present, else main quantile for KPIs
    main_quantile = None
    for col in ["forecast", "forecast_p60", "forecast_p50", "forecast_p90", "forecast_p30", "forecast_p10"]:
        if col in daily_agg.columns:
            main_quantile = col
            break
    if main_quantile:
        forecast_total = daily_agg[main_quantile].sum()
        delta_pct = (forecast_total - actual_total) / actual_total * 100 if actual_total else 0
    else:
        forecast_total = 0
        delta_pct = 0
    # Plot
    raw_numeric_cols: list[str] = []
    selected_raw_metric = (request.query_params.get("raw_metric") or "").strip()
    raw_metric_agg = (request.query_params.get("raw_metric_agg") or "mean").strip().lower()
    raw_metric_axis = (request.query_params.get("raw_metric_axis") or "secondary").strip().lower()
    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        try:
            exclude_cols = set(["date", "sales", "actual", "forecast"])
            exclude_cols.update([str(c) for c in grain if c])
            for c in raw_df.columns:
                if str(c) in exclude_cols:
                    continue
                s = raw_df[c]
                if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                    raw_numeric_cols.append(str(c))
            raw_numeric_cols = sorted(set(raw_numeric_cols))
        except Exception:
            raw_numeric_cols = []

    # Out-of-stock imputation data (reported vs imputed sales)
    oos_imputed = data_store[session_id].get("oos_imputed", False)
    reported_sales_month = None
    try:
        if oos_imputed:
            original_sales_df = data_store[session_id].get("original_sales_df")
            if original_sales_df is not None and not original_sales_df.empty:
                # Process original sales data
                orig_sales = original_sales_df.copy()
                orig_sales["date"] = pd.to_datetime(orig_sales["date"])

                # Apply the same grain filters to original data
                for col in grain:
                    if col in orig_sales.columns and col in selected_grain_values:
                        orig_sales[col] = orig_sales[col].astype(str)
                        orig_sales = orig_sales[orig_sales[col].isin([str(v) for v in selected_grain_values[col]])]

                if not orig_sales.empty and "sales" in orig_sales.columns:
                    orig_sales["sales"] = pd.to_numeric(orig_sales["sales"], errors="coerce").fillna(0)
                    reported_sales_month = (
                        orig_sales.set_index("date")["sales"]
                        .resample("MS")  # Month Start for consistent x-axis alignment
                        .sum()
                        .reset_index()
                    )
    except Exception as e:
        print(f"Warning: could not compute OOS reported sales overlay: {e}")

    decision_period = None
    forecast_for_accuracy = None
    first_forecast_date = None
    try:
        # Find actual/forecast date ranges
        last_actual_date = monthly_agg[monthly_agg['actual'] > 0]['date'].max() if not monthly_agg[monthly_agg['actual'] > 0].empty else monthly_agg['date'].max()
        # Use forecast_p60 as main quantile for accuracy if present, else fallback
        main_quantile = None
        for col in ["forecast_p60", "forecast_p50", "forecast_p90", "forecast_p30", "forecast_p10"]:
            if col in monthly_agg.columns:
                main_quantile = col
                break
        # Use forecast for first_forecast_date if present, else main_quantile
        forecast_for_accuracy = "forecast" if "forecast" in monthly_agg.columns else main_quantile
        first_forecast_date = monthly_agg[monthly_agg[forecast_for_accuracy] > 0]['date'].min() if not monthly_agg[monthly_agg[forecast_for_accuracy] > 0].empty else monthly_agg['date'].min()
        if forecast_for_accuracy:
            decision_source = monthly_agg[monthly_agg[forecast_for_accuracy] > 0]
            if not decision_source.empty:
                decision_period = decision_source['date'].min()
    except Exception as exc:
        decision_period = None

    if decision_period is None and not monthly_agg.empty:
        decision_period = monthly_agg['date'].min()

    if isinstance(decision_period, (pd.Timestamp, datetime)):
        decision_period = decision_period.strftime("%Y-%m")
    else:
        decision_period = str(decision_period) if decision_period else ""

    def calculate_accuracy(row):
        if row['actual'] > 0 and forecast_for_accuracy and row[forecast_for_accuracy] > 0:
            mape = abs(row['actual'] - row[forecast_for_accuracy]) / row['actual'] * 100
            return 100 - mape
        return None

    monthly_agg['accuracy'] = monthly_agg.apply(calculate_accuracy, axis=1)
    monthly_agg_actual = monthly_agg[monthly_agg['date'] <= last_actual_date].copy()
    monthly_agg_forecast = monthly_agg[monthly_agg['date'] >= first_forecast_date].copy()

    latest_sales_date = monthly_agg_actual['date'].max() if not monthly_agg_actual.empty else None
    latest_sales_value = float(monthly_agg_actual[monthly_agg_actual['date'] == latest_sales_date]['actual'].sum()) if latest_sales_date is not None else 0.0
    latest_sales_month = latest_sales_date.strftime("%B %Y") if isinstance(latest_sales_date, pd.Timestamp) else ""

    latest_raw_sales_value = 0.0
    if reported_sales_month is not None and not reported_sales_month.empty and latest_sales_date is not None:
        match = reported_sales_month[reported_sales_month['date'] == latest_sales_date]
        if not match.empty:
            latest_raw_sales_value = float(match['sales'].sum())

    forecast_value_same_month = 0.0
    forecast_month_label = latest_sales_month
    if latest_sales_date is not None and 'forecast' in monthly_agg.columns:
        match_forecast = monthly_agg[monthly_agg['date'] == latest_sales_date]
        if not match_forecast.empty:
            forecast_value_same_month = float(match_forecast['forecast'].sum())
        else:
            latest_forecast_row = monthly_agg_forecast[monthly_agg_forecast['forecast'] > 0]
            if not latest_forecast_row.empty:
                forecast_value_same_month = float(latest_forecast_row.iloc[-1]['forecast'])

    latest_accuracy = None
    accuracy_series = monthly_agg['accuracy'].dropna()
    if not accuracy_series.empty:
        latest_accuracy = float(accuracy_series.iloc[-1])

    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_agg_actual["date"],
            y=monthly_agg_actual["actual"],
            name="Sales (imputed)" if oos_imputed else "Sales",
            mode="lines+markers"
        ))
        if oos_imputed and reported_sales_month is not None and not reported_sales_month.empty:
            fig.add_trace(go.Scatter(
                x=reported_sales_month["date"],
                y=reported_sales_month["sales"],
                name="Sales (reported)",
                mode="lines",
                line=dict(color="rgba(108,117,125,0.85)", dash="dot", width=2),
                opacity=0.9
            ))

        forecast_accuracy_display = monthly_agg_forecast['accuracy'].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
        # Plot original forecast if present
        if "forecast" in monthly_agg_forecast.columns:
            fig.add_trace(go.Scatter(
                x=monthly_agg_forecast["date"],
                y=monthly_agg_forecast["forecast"],
                name="Forecast",
                mode="lines+markers",
                line=dict(color="green", width=2),
                customdata=forecast_accuracy_display.values,
                hovertemplate="Date: %{x|%Y-%m}<br>Forecast Sales: %{y:,.0f}<br>Accuracy: %{customdata}<extra></extra>"
            ))

        # Add quantile lines if present
        quantile_colors = {
            'forecast_p10': 'rgba(255, 99, 132, 0.5)',
            'forecast_p30': 'rgba(255, 206, 86, 0.5)',
            'forecast_p60': 'rgba(54, 162, 235, 0.5)',
            'forecast_p90': 'rgba(75, 192, 192, 0.5)'
        }
        for q in quantile_cols:
            if q in monthly_agg_forecast.columns:
                fig.add_trace(go.Scatter(
                    x=monthly_agg_forecast["date"],
                    y=monthly_agg_forecast[q],
                    name=q.replace("forecast_", "").upper(),
                    mode="lines",
                    line=dict(dash="dot", color=quantile_colors.get(q, None)),
                    opacity=0.8
                ))

        # Optional overlay: numeric column from raw data (monthly aggregation)
        if selected_raw_metric and raw_numeric_cols and selected_raw_metric in raw_numeric_cols and isinstance(raw_df, pd.DataFrame) and "date" in raw_df.columns:
            agg_map = {
                "mean": "mean",
                "sum": "sum",
                "median": "median",
                "min": "min",
                "max": "max",
            }
            agg_func = agg_map.get(raw_metric_agg, "mean")
            axis = "secondary" if raw_metric_axis not in ("primary", "secondary") else raw_metric_axis
            try:
                raw_f = raw_df.copy()
                raw_f["date"] = pd.to_datetime(raw_f["date"])
                for col in grain:
                    if col in raw_f.columns and col in selected_grain_values:
                        raw_f[col] = raw_f[col].astype(str)
                        raw_f = raw_f[raw_f[col].isin([str(v) for v in selected_grain_values[col]])]

                raw_f[selected_raw_metric] = pd.to_numeric(raw_f[selected_raw_metric], errors="coerce")
                raw_f = raw_f.dropna(subset=[selected_raw_metric])
                if not raw_f.empty:
                    raw_month = (
                        raw_f.set_index("date")[selected_raw_metric]
                        .resample("MS")  # Month Start for consistent x-axis alignment
                        .agg(agg_func)
                        .reset_index()
                    )
                    if not raw_month.empty:
                        trace_name = f"{selected_raw_metric} ({agg_func})"
                        color = "rgba(111, 66, 193, 0.9)"
                        if axis == "secondary":
                            fig.add_trace(go.Scatter(
                                x=raw_month["date"],
                                y=raw_month[selected_raw_metric],
                                name=trace_name,
                                mode="lines+markers",
                                line=dict(color=color, width=2),
                                marker=dict(size=6, color=color),
                                yaxis="y2",
                                opacity=0.95
                            ))
                            fig.update_layout(
                                yaxis2=dict(
                                    title=selected_raw_metric,
                                    overlaying="y",
                                    side="right",
                                    showgrid=False,
                                    zeroline=False
                                )
                            )
                        else:
                            fig.add_trace(go.Scatter(
                                x=raw_month["date"],
                                y=raw_month[selected_raw_metric],
                                name=trace_name,
                                mode="lines+markers",
                                line=dict(color=color, width=2),
                                marker=dict(size=6, color=color),
                                opacity=0.95
                            ))
            except Exception as e:
                print(f"Warning: Error creating raw metric overlay plot: {e}")

        fig.update_layout(
            template="plotly_white",
            height=450,
            xaxis_title="Month",
            yaxis_title="Sales",
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#23243a",
                bordercolor="#4e6cf4",
                font=dict(family="Inter, sans-serif", size=14, color="#fff"),
                namelength=-1
            ),
            xaxis=dict(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor="#4e6cf4",
                spikethickness=2,
                spikedash="solid",
                showline=True,
                showgrid=True,
                zeroline=False,
                showticklabels=True
            ),
            yaxis=dict(
                showspikes=False
            )
        )
        plot_html = fig.to_html(full_html=False)
    except Exception as e:
        # Fallback simple plot if there's any issue
        print(f"Warning: Error creating main plot: {e}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_agg["date"],
            y=monthly_agg["actual"],
            name="Actual",
            mode="lines+markers"
        ))
        fig.add_trace(go.Scatter(
            x=monthly_agg["date"],
            y=monthly_agg["forecast"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color="green", width=2)
        ))
        plot_html = fig.to_html(full_html=False)
    
    # Drivers (SHAP-based summaries) from forecast artifacts
    driver_artifacts = data_store.get(session_id, {}).get("driver_artifacts") or {}
    directional_view = []
    for row in (driver_artifacts.get("directional") or []):
        try:
            directional_view.append({
                "feature": row.get("feature"),
                "label": _humanize_feature_name(row.get("feature")),
                "effect": row.get("effect") or "Low",
                "direction": row.get("direction") or "Mixed",
                "strength_norm": float(row.get("strength_norm") or 0.0),
            })
        except Exception:
            continue

    local_drivers = []
    local_driver_meta = None
    local_driver_error = None
    try:
        driver_model = driver_artifacts.get("model")
        driver_features = driver_artifacts.get("features") or []
        raw_df = data_store.get(session_id, {}).get("df")

        is_single_series = True
        series_key = {}
        for col in grain:
            if col not in filtered_df.columns:
                continue
            uniq = filtered_df[col].dropna().astype(str).unique()
            if len(uniq) != 1:
                is_single_series = False
                break
            series_key[col] = str(uniq[0])

        forecast_col_for_drivers = None
        for col in ["forecast", "forecast_p60", "forecast_p50", "forecast_p90", "forecast_p30", "forecast_p10"]:
            if col in filtered_df.columns:
                forecast_col_for_drivers = col
                break

        start_month = data_store.get(session_id, {}).get("start_month")
        start_dt = pd.to_datetime(f"{start_month}-01") if start_month else None

        driver_date = None
        if is_single_series and forecast_col_for_drivers and start_dt is not None:
            cand = filtered_df[(filtered_df["date"] >= start_dt) & (filtered_df[forecast_col_for_drivers].fillna(0) > 0)]
            if not cand.empty:
                driver_date = pd.to_datetime(cand["date"].min())

        if not is_single_series:
            local_driver_error = "Select a single series in filters to see local drivers."
        elif driver_model is None or not driver_features or raw_df is None or not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
            local_driver_error = "Drivers are not available yet (rerun forecast to enable drivers)."
        elif driver_date is None:
            local_driver_error = "No forecast point found to explain for the current selection."
        else:
            from features import add_features as _add_features, get_feature_columns as _get_feature_columns
            import config as _config
            import numpy as np
            import xgboost as xgb

            df_series = raw_df.copy()
            df_series["date"] = pd.to_datetime(df_series["date"])
            mask = pd.Series([True] * len(df_series))
            for col, val in series_key.items():
                if col in df_series.columns:
                    mask &= df_series[col].astype(str) == str(val)
            df_series = df_series[mask].sort_values("date")
            if df_series.empty:
                raise ValueError("No raw rows found for selected series.")

            df_hist = df_series[df_series["date"] < driver_date].copy()
            if df_hist.empty:
                raise ValueError("Not enough history before the explained date.")

            last = df_hist.iloc[-1].to_dict()
            new_row = dict(last)
            new_row["date"] = driver_date
            if "sales" in new_row:
                new_row["sales"] = 0
            df_feat_in = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)

            df_feat = _add_features(df_feat_in, seasonal_flags=_config.SEASONAL_FLAGS)
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
            _, cat_cols = _get_feature_columns(df_feat)
            existing_cat = [c for c in cat_cols if c in df_feat.columns]
            if existing_cat:
                df_feat = pd.get_dummies(df_feat, columns=existing_cat, drop_first=True, dtype=int)
            for f in driver_features:
                if f not in df_feat.columns:
                    df_feat[f] = 0

            row = df_feat[df_feat["date"] == driver_date]
            if row.empty:
                raise ValueError("Could not build feature row for explained date.")
            X = row[driver_features].iloc[-1:]

            booster = driver_model.get_booster()
            dm = xgb.DMatrix(X, feature_names=driver_features)
            contrib = booster.predict(dm, pred_contribs=True)[0]
            bias = float(contrib[-1])
            shap_vals = contrib[:-1]
            pred_raw = float(bias + float(np.sum(shap_vals)))
            pred_display = max(0.0, pred_raw)

            pairs = list(zip(driver_features, shap_vals, X.iloc[0].tolist()))
            pairs.sort(key=lambda t: abs(float(t[1])), reverse=True)
            top = pairs[:8]
            for feat, sv, val in top:
                sv_f = float(sv)
                direction = "↑" if sv_f > 0 else ("↓" if sv_f < 0 else "—")
                pct = (sv_f / pred_raw * 100.0) if abs(pred_raw) > 1e-9 else None
                local_drivers.append({
                    "feature": feat,
                    "label": _humanize_feature_name(feat),
                    "value": float(val) if isinstance(val, (int, float, np.floating, np.integer)) else val,
                    "contribution": sv_f,
                    "direction": direction,
                    "pct": pct,
                })

            local_driver_meta = {
                "date": driver_date.strftime("%Y-%m-%d"),
                "forecast_col": forecast_col_for_drivers,
                "prediction": pred_display,
                "base": bias,
            }
    except Exception as e:
        local_driver_error = f"Local drivers unavailable: {e}"
    
    # CSV
    csv_data = filtered_df.to_csv(index=False)
    # Prepare quantile columns for table/plot
    quantile_cols = [col for col in filtered_df.columns if col.startswith("forecast_p")]
    return templates.TemplateResponse("results_v2.html", {
        "request": request,
        "actual_total": f"{actual_total:,.0f}",
        "forecast_total": f"{forecast_total:,.0f}",
        "delta_pct": f"{delta_pct:.1f}%",
        "plot_html": plot_html,
        "directional_view": directional_view,
        "local_drivers": local_drivers,
        "local_driver_meta": local_driver_meta,
        "local_driver_error": local_driver_error,
        "raw_numeric_cols": raw_numeric_cols,
        "selected_raw_metric": selected_raw_metric,
        "raw_metric_agg": raw_metric_agg,
        "raw_metric_axis": raw_metric_axis,
        "oos_imputed": oos_imputed,
        "csv_data": csv_data,
        "grain": grain,
        "all_grain_values": all_grain_values,
        "selected_grain_values": selected_grain_values,
        "current_date": datetime.now().strftime("%B %d, %Y"),
        "forecast_start_month": data_store.get(session_id, {}).get("start_month"),
        "forecast_months": data_store.get(session_id, {}).get("months"),
        "latest_sales_month": latest_sales_month,
        "latest_sales_value": latest_sales_value,
        "latest_raw_sales_value": latest_raw_sales_value,
        "forecast_month_label": forecast_month_label,
        "latest_forecast_value": forecast_value_same_month,
        "latest_accuracy": latest_accuracy,
        "quantile_cols": quantile_cols,
        "filtered_df": filtered_df
    })

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    # Backward compatibility: dashboard now owns history browsing.
    user_email = _get_user_email(request)
    if not user_email:
        return RedirectResponse("/signin?next=/dashboard", status_code=303)
    return RedirectResponse("/dashboard", status_code=303)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    user_email = _get_user_email(request)
    if not user_email:
        return RedirectResponse("/signin?next=/dashboard", status_code=303)
    try:
        import history_store
        runs = history_store.list_forecast_runs(user_email, limit=100)
        history_backend = history_store.history_backend()
        history_info = history_store.history_connection_info()
    except Exception as e:
        logging.warning(f"[HISTORY] Failed to list dashboard runs for {user_email}: {e}")
        runs = []
        history_backend = "unknown"
        history_info = {"backend": "unknown"}
    session_id = _session_id_from_request(request)
    history_error = None
    try:
        history_error = data_store.get(session_id, {}).get("history_last_error")
    except Exception:
        history_error = None

    running_forecast = None
    running_meta = {}
    try:
        s = data_store.get(session_id, {}) or {}
        prog = s.get("forecast_progress") or None
        if isinstance(prog, dict) and prog and not bool(prog.get("done", False)) and not bool(prog.get("cancelled", False)):
            running_forecast = prog
            running_meta = {
                "uploaded_filename": s.get("uploaded_filename"),
                "start_month": s.get("start_month"),
                "months": s.get("months"),
                "grain": s.get("grain"),
            }
    except Exception:
        running_forecast = None
        running_meta = {}
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "runs": runs,
            "history_backend": history_backend,
            "history_info": history_info,
            "history_error": history_error,
            "running_forecast": running_forecast,
            "running_meta": running_meta,
        },
    )

@app.post("/history/load")
async def history_load(request: Request, run_id: int = Form(...), target: str = Form(default="results")):
    user_email = _get_user_email(request)
    if not user_email:
        return RedirectResponse("/signin?next=/history", status_code=303)

    import history_store
    session_id = _session_id_from_request(request)
    run = history_store.load_forecast_run(user_email, int(run_id))

    # Rehydrate session state so existing pages keep working without logic changes.
    session = data_store.setdefault(session_id, {})
    raw_df = run.get("raw_df") if isinstance(run, dict) else None
    forecast_df = run.get("forecast_df") if isinstance(run, dict) else None
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        raise HTTPException(status_code=400, detail="Saved run is missing raw data.")
    if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
        raise HTTPException(status_code=400, detail="Saved run is missing forecast output.")

    # Parse expected date columns for downstream plotting/resampling.
    if "date" in raw_df.columns:
        raw_df = raw_df.copy()
        raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    if "date" in forecast_df.columns:
        forecast_df = forecast_df.copy()
        forecast_df["date"] = pd.to_datetime(forecast_df["date"], errors="coerce")

    params = run.get("params") if isinstance(run, dict) else {}
    if not isinstance(params, dict):
        params = {}

    session["df"] = raw_df
    session["raw_df"] = raw_df.copy()
    session["forecast_df"] = forecast_df
    session["feature_importance"] = run.get("feature_importance") if isinstance(run.get("feature_importance"), dict) else {}
    session["driver_artifacts"] = run.get("driver_artifacts") if isinstance(run.get("driver_artifacts"), dict) else {}
    session["uploaded_filename"] = run.get("filename")
    session["start_month"] = params.get("start_month")
    session["months"] = params.get("months")
    session["grain"] = params.get("grain") or session.get("grain") or ["item", "store"]
    session["extra_features"] = params.get("extra_features") or session.get("extra_features") or []
    session["oos_enabled"] = bool(params.get("oos_enabled") or False)
    session["oos_column"] = params.get("oos_column")
    session["forecast_run_id"] = int(run_id)

    if str(target or "").lower() == "supply_plan":
        try:
            sp = history_store.load_supply_plan(user_email, int(run_id))
            sp_df = sp.get("supply_plan_df")
            sp_full = sp.get("supply_plan_full_df")
            if isinstance(sp_df, pd.DataFrame) and not sp_df.empty:
                if "period_start" in sp_df.columns:
                    sp_df = sp_df.copy()
                    sp_df["period_start"] = pd.to_datetime(sp_df["period_start"], errors="coerce")
                session["supply_plan_df"] = sp_df
            if isinstance(sp_full, pd.DataFrame) and not sp_full.empty:
                if "period_start" in sp_full.columns:
                    sp_full = sp_full.copy()
                    sp_full["period_start"] = pd.to_datetime(sp_full["period_start"], errors="coerce")
                session["supply_plan_full_df"] = sp_full
        except KeyError:
            raise HTTPException(status_code=404, detail="No saved supply plan for this run.")
        return RedirectResponse("/supply_plan", status_code=303)

    return RedirectResponse("/results", status_code=303)

@app.get("/api/update_plot")
async def api_update_plot(request: Request):
    """AJAX endpoint to update plot and KPIs without full page reload"""
    session_id = _session_id_from_request(request)
    if "forecast_df" not in data_store.get(session_id, {}):
        return JSONResponse({"error": "No forecast available"}, status_code=400)

    forecast_df = data_store[session_id]["forecast_df"]
    feature_importance = data_store[session_id].get("feature_importance", {})
    grain = data_store[session_id].get("grain", ["item", "store"])
    raw_df = data_store.get(session_id, {}).get("df")

    if forecast_df.empty:
        return JSONResponse({"error": "Forecast data is empty"}, status_code=400)

    # Build dynamic filters for each grain column
    all_grain_values = {}
    selected_grain_values = {}
    filter_mask = pd.Series([True] * len(forecast_df))

    for col in grain:
        if col in forecast_df.columns:
            non_nan_vals = forecast_df[col][~forecast_df[col].isna()]
            all_vals = sorted(non_nan_vals.astype(str).unique())
            forecast_df[col] = forecast_df[col].astype(str)
            selected_vals = request.query_params.getlist(col) or all_vals
            all_grain_values[col] = all_vals
            selected_grain_values[col] = selected_vals
            filter_mask &= forecast_df[col].isin(selected_vals)

    filtered_df = forecast_df[filter_mask].copy()
    if filtered_df.empty:
        return JSONResponse({"error": "No data matches the selected filters"}, status_code=400)

    # Aggregate
    groupby_cols = [col for col in ["date"] + grain if col in filtered_df.columns]
    quantile_cols = [col for col in filtered_df.columns if col.startswith("forecast_p")]
    agg_dict = {"actual": "sum"}
    if "forecast" in filtered_df.columns:
        agg_dict["forecast"] = "sum"
    for q in quantile_cols:
        agg_dict[q] = "sum"

    daily_agg = (
        filtered_df
        .groupby(groupby_cols, as_index=False)
        .agg(agg_dict)
    )
    monthly_agg = (
        daily_agg
        .set_index("date")
        .resample("MS")
        .sum()
        .reset_index()
    )

    # KPIs
    actual_total = daily_agg["actual"].sum()
    main_quantile = None
    for col in ["forecast", "forecast_p60", "forecast_p50", "forecast_p90", "forecast_p30", "forecast_p10"]:
        if col in daily_agg.columns:
            main_quantile = col
            break
    if main_quantile:
        forecast_total = daily_agg[main_quantile].sum()
        delta_pct = (forecast_total - actual_total) / actual_total * 100 if actual_total else 0
    else:
        forecast_total = 0
        delta_pct = 0

    # Raw metric overlay data
    raw_numeric_cols: list[str] = []
    selected_raw_metric = (request.query_params.get("raw_metric") or "").strip()
    raw_metric_agg = (request.query_params.get("raw_metric_agg") or "mean").strip().lower()
    raw_metric_axis = (request.query_params.get("raw_metric_axis") or "secondary").strip().lower()
    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        try:
            exclude_cols = set(["date", "sales", "actual", "forecast"])
            exclude_cols.update([str(c) for c in grain if c])
            for c in raw_df.columns:
                if str(c) in exclude_cols:
                    continue
                s = raw_df[c]
                if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                    raw_numeric_cols.append(str(c))
            raw_numeric_cols = sorted(set(raw_numeric_cols))
        except Exception:
            raw_numeric_cols = []

    # Out-of-stock imputation data (reported vs imputed sales)
    oos_imputed = data_store[session_id].get("oos_imputed", False)
    reported_sales_month = None
    try:
        if oos_imputed:
            original_sales_df = data_store[session_id].get("original_sales_df")
            if original_sales_df is not None and not original_sales_df.empty:
                # Process original sales data
                orig_sales = original_sales_df.copy()
                orig_sales["date"] = pd.to_datetime(orig_sales["date"])

                # Apply the same grain filters to original data
                for col in grain:
                    if col in orig_sales.columns and col in selected_grain_values:
                        orig_sales[col] = orig_sales[col].astype(str)
                        orig_sales = orig_sales[orig_sales[col].isin([str(v) for v in selected_grain_values[col]])]

                if not orig_sales.empty and "sales" in orig_sales.columns:
                    orig_sales["sales"] = pd.to_numeric(orig_sales["sales"], errors="coerce").fillna(0)
                    reported_sales_month = (
                        orig_sales.set_index("date")["sales"]
                        .resample("MS")  # Month Start for consistent x-axis alignment
                        .sum()
                        .reset_index()
                    )
    except Exception as e:
        print(f"Warning: could not compute OOS reported sales overlay: {e}")

    try:
        # Create plot
        last_actual_date = monthly_agg[monthly_agg['actual'] > 0]['date'].max() if not monthly_agg[monthly_agg['actual'] > 0].empty else monthly_agg['date'].max()
        main_quantile = None
        for col in ["forecast_p60", "forecast_p50", "forecast_p90", "forecast_p30", "forecast_p10"]:
            if col in monthly_agg.columns:
                main_quantile = col
                break
        forecast_for_accuracy = "forecast" if "forecast" in monthly_agg.columns else main_quantile
        first_forecast_date = monthly_agg[monthly_agg[forecast_for_accuracy] > 0]['date'].min() if not monthly_agg[monthly_agg[forecast_for_accuracy] > 0].empty else monthly_agg['date'].min()

        def calculate_accuracy(row):
            if row['actual'] > 0 and forecast_for_accuracy and row[forecast_for_accuracy] > 0:
                mape = abs(row['actual'] - row[forecast_for_accuracy]) / row['actual'] * 100
                return 100 - mape
            return None

        monthly_agg['accuracy'] = monthly_agg.apply(calculate_accuracy, axis=1)
        monthly_agg_actual = monthly_agg[monthly_agg['date'] <= last_actual_date].copy()
        monthly_agg_forecast = monthly_agg[monthly_agg['date'] >= first_forecast_date].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_agg_actual["date"],
            y=monthly_agg_actual["actual"],
            name="Sales (imputed)" if oos_imputed else "Sales",
            mode="lines+markers"
        ))
        if oos_imputed and reported_sales_month is not None and not reported_sales_month.empty:
            fig.add_trace(go.Scatter(
                x=reported_sales_month["date"],
                y=reported_sales_month["sales"],
                name="Sales (reported)",
                mode="lines",
                line=dict(color="rgba(108,117,125,0.85)", dash="dot", width=2),
                opacity=0.9
            ))

        forecast_accuracy_display = monthly_agg_forecast['accuracy'].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
        if "forecast" in monthly_agg_forecast.columns:
            fig.add_trace(go.Scatter(
                x=monthly_agg_forecast["date"],
                y=monthly_agg_forecast["forecast"],
                name="Forecast",
                mode="lines+markers",
                line=dict(color="green", width=2),
                customdata=forecast_accuracy_display.values,
                hovertemplate="Date: %{x|%Y-%m}<br>Forecast Sales: %{y:,.0f}<br>Accuracy: %{customdata}<extra></extra>"
            ))

        quantile_colors = {
            'forecast_p10': 'rgba(255, 99, 132, 0.5)',
            'forecast_p30': 'rgba(255, 206, 86, 0.5)',
            'forecast_p60': 'rgba(54, 162, 235, 0.5)',
            'forecast_p90': 'rgba(75, 192, 192, 0.5)'
        }
        for q in quantile_cols:
            if q in monthly_agg_forecast.columns:
                fig.add_trace(go.Scatter(
                    x=monthly_agg_forecast["date"],
                    y=monthly_agg_forecast[q],
                    name=q.replace("forecast_", "").upper(),
                    mode="lines",
                    line=dict(dash="dot", color=quantile_colors.get(q, None)),
                    opacity=0.8
                ))

        # Optional overlay: numeric column from raw data
        if selected_raw_metric and raw_numeric_cols and selected_raw_metric in raw_numeric_cols and isinstance(raw_df, pd.DataFrame) and "date" in raw_df.columns:
            agg_map = {"mean": "mean", "sum": "sum", "median": "median", "min": "min", "max": "max"}
            agg_func = agg_map.get(raw_metric_agg, "mean")
            axis = "secondary" if raw_metric_axis not in ("primary", "secondary") else raw_metric_axis
            try:
                raw_f = raw_df.copy()
                raw_f["date"] = pd.to_datetime(raw_f["date"])
                for col in grain:
                    if col in raw_f.columns and col in selected_grain_values:
                        raw_f[col] = raw_f[col].astype(str)
                        raw_f = raw_f[raw_f[col].isin([str(v) for v in selected_grain_values[col]])]

                raw_f[selected_raw_metric] = pd.to_numeric(raw_f[selected_raw_metric], errors="coerce")
                raw_f = raw_f.dropna(subset=[selected_raw_metric])
                if not raw_f.empty:
                    raw_month = (
                        raw_f.set_index("date")[selected_raw_metric]
                        .resample("MS")
                        .agg(agg_func)
                        .reset_index()
                    )
                    if not raw_month.empty:
                        trace_name = f"{selected_raw_metric} ({agg_func})"
                        color = "rgba(111, 66, 193, 0.9)"
                        if axis == "secondary":
                            fig.add_trace(go.Scatter(
                                x=raw_month["date"],
                                y=raw_month[selected_raw_metric],
                                name=trace_name,
                                mode="lines+markers",
                                line=dict(color=color, width=2),
                                marker=dict(size=6, color=color),
                                yaxis="y2",
                                opacity=0.95
                            ))
                            fig.update_layout(
                                yaxis2=dict(
                                    title=selected_raw_metric,
                                    overlaying="y",
                                    side="right",
                                    showgrid=False,
                                    zeroline=False
                                )
                            )
                        else:
                            fig.add_trace(go.Scatter(
                                x=raw_month["date"],
                                y=raw_month[selected_raw_metric],
                                name=trace_name,
                                mode="lines+markers",
                                line=dict(color=color, width=2),
                                marker=dict(size=6, color=color),
                                opacity=0.95
                            ))
            except Exception as e:
                print(f"Warning: Error creating raw metric overlay plot: {e}")

        fig.update_layout(
            template="plotly_white",
            height=450,
            xaxis_title="Month",
            yaxis_title="Sales",
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#23243a",
                bordercolor="#4e6cf4",
                font=dict(family="Inter, sans-serif", size=14, color="#fff"),
                namelength=-1
            ),
            xaxis=dict(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor="#4e6cf4",
                spikethickness=2,
                spikedash="solid",
                showline=True,
                showgrid=True,
                zeroline=False,
                showticklabels=True
            ),
            yaxis=dict(
                showspikes=False
            )
        )
        # Return plot as JSON for Plotly.react() (same as supply plan page)
        # Use Plotly's to_json() to properly handle numpy arrays
        plot_json = fig.to_json()
    except Exception as e:
        print(f"Error creating plot: {e}")
        return JSONResponse({"error": f"Error creating plot: {str(e)}"}, status_code=500)

    return JSONResponse({
        "success": True,
        "plot_json": plot_json,
        "actual_total": f"{actual_total:,.0f}",
        "forecast_total": f"{forecast_total:,.0f}",
        "delta_pct": f"{delta_pct:.1f}%",
        "filtered_rows": len(filtered_df)
    })


# --- Supply Planning Endpoints ---
from fastapi import Form

@app.get("/supply_plan", response_class=HTMLResponse)
async def supply_plan_page(request: Request):
    logging.debug("[LOG] /supply_plan GET endpoint called")
    session_id = _session_id_from_request(request)
    forecast_df = data_store.get(session_id, {}).get("forecast_df")
    has_forecast = forecast_df is not None and isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty
    raw_df = data_store.get(session_id, {}).get("df")
    has_raw_data = raw_df is not None and isinstance(raw_df, pd.DataFrame) and not raw_df.empty
    forecast_start_month = data_store.get(session_id, {}).get("start_month")
    forecast_months = data_store.get(session_id, {}).get("months")

    grain_cols = []
    if has_forecast:
        exclude = {"date", "actual", "forecast"}
        grain_cols = [
            c for c in forecast_df.columns
            if c not in exclude and not str(c).startswith("forecast_p")
        ]

    grain_label = ", ".join([str(c) for c in grain_cols]) if grain_cols else "item, store"

    combos = []
    if has_forecast:
        df = forecast_df.copy()
        sku_col = "item" if "item" in df.columns else (grain_cols[0] if len(grain_cols) >= 1 else None)
        loc_col = "store" if "store" in df.columns else (grain_cols[1] if len(grain_cols) >= 2 else None)
        extra_cols = [c for c in grain_cols if c not in {sku_col, loc_col} and c in df.columns]

        if sku_col is not None:
            df["sku_id"] = df[sku_col].astype(str)
        else:
            df["sku_id"] = "ALL"

        if loc_col is not None:
            df["location"] = df[loc_col].astype(str)
        else:
            df["location"] = "ALL"

        for col in extra_cols:
            df["location"] = df["location"] + "|" + str(col) + "=" + df[col].astype(str)

        base = df[[*(c for c in [sku_col, loc_col, *extra_cols] if c), "sku_id", "location"]].drop_duplicates()
        base = base.head(200)
        combos = [
            {
                "key": f"{row['sku_id']}|||{row['location']}",
                "label": " / ".join([str(row[c]) for c in [sku_col, loc_col, *extra_cols] if c]) if any([sku_col, loc_col, *extra_cols]) else f"{row['sku_id']} @ {row['location']}",
                "sku_id": str(row["sku_id"]),
                "location": str(row["location"]),
            }
            for _, row in base.iterrows()
        ]

    if has_forecast:
        # When forecast exists, default to auto-fill (so the user doesn't accidentally submit example master data).
        sample_forecast_csv = ""
        sample_inventory_csv = ""
        sample_constraints_csv = ""
        sample_policy_csv = ""
    else:
        sample_forecast_csv = (
            "sku_id,location,week_start,forecast_demand\n"
            "SKU_001,Mumbai,2026-02-03,420\n"
            "SKU_001,Mumbai,2026-02-10,460\n"
            "SKU_001,Mumbai,2026-02-17,480\n"
            "SKU_002,Delhi,2026-02-03,300\n"
        )
        sample_inventory_csv = (
            "sku_id,location,on_hand,allocated,backorders\n"
            "SKU_001,Mumbai,520,80,0\n"
            "SKU_002,Delhi,150,30,20\n"
        )
        sample_constraints_csv = (
            "sku_id,supplier,lead_time_days,moq,order_multiple,max_capacity_per_week,shelf_life_days\n"
            "SKU_001,SUP_A,14,200,20,2000,180\n"
            "SKU_002,SUP_B,21,100,10,1200,90\n"
        )
        sample_policy_csv = (
            "sku_id,holding_cost_per_unit,stockout_cost_per_unit,service_level\n"
            "SKU_001,3.5,45,0.95\n"
            "SKU_002,2.0,30,0.90\n"
        )
    return templates.TemplateResponse(
        "supply_plan.html",
        {
            "request": request,
            "has_forecast": has_forecast,
            "has_raw_data": has_raw_data,
            "grain_label": grain_label,
            "combos": combos,
            "forecast_start_month": forecast_start_month,
            "forecast_months": forecast_months,
            "sample_forecast_csv": sample_forecast_csv,
            "sample_inventory_csv": sample_inventory_csv,
            "sample_constraints_csv": sample_constraints_csv,
            "sample_policy_csv": sample_policy_csv,
            "raw_inventory_columns": ([
                c for c in (list(raw_df.columns) if has_raw_data else [])
                if any(k in str(c).lower() for k in ["on_hand", "onhand", "inventory", "stock", "qty_on_hand", "qoh", "available"]) 
            ] if has_raw_data else []),
            "detected_inventory_col": (next((c for c in (list(raw_df.columns) if has_raw_data else []) if any(k in str(c).lower() for k in ["on_hand", "onhand", "inventory", "stock", "qty_on_hand", "qoh", "available"])), None) if has_raw_data else None),
        }
    )

from fastapi import Body

@app.get("/supply_plan_defaults")
async def supply_plan_defaults(request: Request, combo_key: str):
    """
    Returns default inputs for a selected series so the UI can prefill editable fields.
    """
    session_id = _session_id_from_request(request)
    forecast_df = data_store.get(session_id, {}).get("forecast_df")
    if forecast_df is None or not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
        raise HTTPException(status_code=400, detail="No forecast data available.")

    start_month = data_store.get(session_id, {}).get("start_month")
    months_val = data_store.get(session_id, {}).get("months")
    if not start_month or not months_val:
        raise HTTPException(status_code=400, detail="Forecast start month / horizon not found.")

    try:
        months_val = int(months_val)
        months_val = max(1, min(months_val, 120))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid forecast horizon.")

    start_ts = pd.to_datetime(f"{start_month}-01", errors="coerce")
    if pd.isna(start_ts):
        raise HTTPException(status_code=400, detail="Invalid forecast start month.")
    end_ts = start_ts + pd.DateOffset(months=months_val)
    months_index = pd.date_range(start=start_ts, periods=months_val, freq="MS")

    selected_sku = None
    selected_location = None
    if isinstance(combo_key, str) and "|||" in combo_key:
        parts = combo_key.split("|||", 1)
        selected_sku = parts[0]
        selected_location = parts[1]
    if not selected_sku or selected_location is None:
        raise HTTPException(status_code=400, detail="Invalid combo_key.")

    # Build sku/location keys in the same way as supply planning
    df = forecast_df.copy()
    if "forecast" not in df.columns:
        quantile_fallback = next((c for c in ["forecast_p60", "forecast_p50"] if c in df.columns), None)
        if not quantile_fallback:
            raise HTTPException(status_code=400, detail="Latest forecast missing forecast column.")
        df = df.rename(columns={quantile_fallback: "forecast"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= start_ts) & (df["date"] < end_ts)]

    exclude = {"date", "actual", "forecast"}
    grain_cols = [c for c in df.columns if c not in exclude and not str(c).startswith("forecast_p")]
    sku_col = "item" if "item" in df.columns else (grain_cols[0] if len(grain_cols) >= 1 else None)
    loc_col = "store" if "store" in df.columns else (grain_cols[1] if len(grain_cols) >= 2 else None)
    extra_cols = [c for c in grain_cols if c not in {sku_col, loc_col} and c in df.columns]

    df["sku_id"] = df[sku_col].astype(str) if sku_col else "ALL"
    df["location"] = df[loc_col].astype(str) if loc_col else "ALL"
    for col in extra_cols:
        df["location"] = df["location"] + "|" + str(col) + "=" + df[col].astype(str)

    df = df[(df["sku_id"].astype(str) == str(selected_sku)) & (df["location"].astype(str) == str(selected_location))]
    if df.empty:
        raise HTTPException(status_code=404, detail="Selected series not found in forecast data.")

    df["period_start"] = df["date"].dt.to_period("M").apply(lambda p: p.start_time)
    df["forecast_demand"] = pd.to_numeric(df["forecast"], errors="coerce").fillna(0.0)

    monthly = df.groupby(["sku_id", "location", "period_start"], as_index=False)["forecast_demand"].sum()
    grid = pd.DataFrame({"period_start": months_index}).merge(monthly, on="period_start", how="left")
    grid["sku_id"] = str(selected_sku)
    grid["location"] = str(selected_location)
    grid["forecast_demand"] = grid["forecast_demand"].fillna(0.0)

    mean_monthly = float(grid["forecast_demand"].mean() or 0.0)
    import hashlib
    import numpy as np

    def _seed(key: str) -> int:
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    rng_c = np.random.default_rng(_seed(f"constraints:{selected_sku}"))
    lead_time_days = int(rng_c.choice([7, 14, 21, 28]))
    moq = int(rng_c.choice([50, 100, 200, 300]))
    order_multiple = int(rng_c.choice([5, 10, 20, 25]))
    cap = float(max(200.0, mean_monthly * rng_c.uniform(3.0, 8.0)))
    max_capacity_per_week = int(np.ceil((cap / 4.0) / order_multiple) * order_multiple)

    rng_p = np.random.default_rng(_seed(f"policy:{selected_sku}"))
    service_level = float(rng_p.choice([0.90, 0.92, 0.95, 0.97]))
    holding_cost = round(float(rng_p.uniform(0.5, 6.0)), 2)
    stockout_cost = round(float(rng_p.uniform(10.0, 80.0)), 2)

    rng_i = np.random.default_rng(_seed(f"inv:{selected_sku}:{selected_location}"))
    on_hand = int(np.ceil(max(0.0, mean_monthly * rng_i.uniform(0.5, 3.0))))
    allocated = int(np.floor(max(0.0, on_hand * rng_i.uniform(0.0, 0.15))))
    backorders = int(np.ceil(max(0.0, mean_monthly * rng_i.uniform(0.0, 0.25))))

    forecast_csv = "period_start,forecast_demand\n" + "\n".join(
        f"{pd.to_datetime(r['period_start']).strftime('%Y-%m-%d')},{float(r['forecast_demand']):.0f}"
        for _, r in grid.iterrows()
    )

    return {
        "constraints": {
            "lead_time_days": lead_time_days,
            "moq": moq,
            "order_multiple": order_multiple,
            "max_capacity_per_week": max_capacity_per_week,
        },
        "policy": {
            "service_level": service_level,
            "holding_cost_per_unit": holding_cost,
            "stockout_cost_per_unit": stockout_cost,
        },
        "inventory": {
            "on_hand": on_hand,
            "allocated": allocated,
            "backorders": backorders,
        },
        "forecast_monthly_csv": forecast_csv,
    }

@app.post("/supply_plan")
async def supply_plan_submit(
    request: Request,
    payload: Dict = Body(...)
):
    logging.debug("[LOG] /supply_plan POST endpoint called")
    session_id = _session_id_from_request(request)
    user_email = _get_user_email(request)
    forecast_df = data_store.get(session_id, {}).get("forecast_df")
    try:
        from io import StringIO
        import json
        import hashlib
        import numpy as np

        def _read_csv_optional(text: str, name: str) -> pd.DataFrame | None:
            if not isinstance(text, str) or not text.strip():
                return None
            try:
                return pd.read_csv(StringIO(text.strip()))
            except Exception as e:
                raise ValueError(f"Failed to parse {name}: {e}")

        def _stable_seed(value: str) -> int:
            h = hashlib.md5(value.encode("utf-8")).hexdigest()
            return int(h[:8], 16)

        def _infer_grain_cols_from_forecast_df(df: pd.DataFrame) -> list[str]:
            exclude = {"date", "actual", "forecast"}
            cols = [c for c in df.columns if c not in exclude and not str(c).startswith("forecast_p")]
            return [str(c) for c in cols]

        def _build_sku_location_columns(df: pd.DataFrame, grain_cols: list[str]) -> tuple[pd.DataFrame, str | None, str | None, list[str]]:
            """
            Convert PiTensor grain columns into supply-planning keys:
            - sku_id (prefer 'item')
            - location (prefer 'store', plus any extra grain columns for uniqueness)
            """
            work = df.copy()
            sku_col = "item" if "item" in work.columns else (grain_cols[0] if len(grain_cols) >= 1 else None)
            loc_col = "store" if "store" in work.columns else (grain_cols[1] if len(grain_cols) >= 2 else None)
            extra_cols = [c for c in grain_cols if c not in {sku_col, loc_col} and c in work.columns]

            if sku_col is None:
                work["sku_id"] = "ALL"
            else:
                work["sku_id"] = work[sku_col].astype(str)

            if loc_col is None:
                work["location"] = "ALL"
            else:
                work["location"] = work[loc_col].astype(str)

            for col in extra_cols:
                work["location"] = work["location"] + "|" + col + "=" + work[col].astype(str)

            return work, sku_col, loc_col, extra_cols

        def _generate_constraints_df(forecast_weekly: pd.DataFrame) -> pd.DataFrame:
            sku_ids = sorted(forecast_weekly["sku_id"].astype(str).unique().tolist())
            mean_weekly = forecast_weekly.groupby("sku_id")["forecast_demand"].mean().to_dict()
            rows = []
            for sku in sku_ids:
                rng = np.random.default_rng(_stable_seed(f"constraints:{sku}"))
                lead_time_days = int(rng.choice([7, 14, 21, 28]))
                moq = int(rng.choice([50, 100, 200, 300]))
                order_multiple = int(rng.choice([5, 10, 20, 25]))
                cap = float(max(200.0, (mean_weekly.get(sku, 0.0) or 0.0) * rng.uniform(3.0, 8.0)))
                rows.append({
                    "sku_id": sku,
                    "supplier": f"SUP_{int(rng.integers(1, 6))}",
                    "lead_time_days": lead_time_days,
                    "moq": moq,
                    "order_multiple": order_multiple,
                    "max_capacity_per_week": int(np.ceil(cap / order_multiple) * order_multiple),
                    "shelf_life_days": int(rng.choice([60, 90, 120, 180])),
                })
            return pd.DataFrame(rows)

        def _generate_policy_df(constraints_df: pd.DataFrame) -> pd.DataFrame:
            sku_ids = sorted(constraints_df["sku_id"].astype(str).unique().tolist())
            rows = []
            for sku in sku_ids:
                rng = np.random.default_rng(_stable_seed(f"policy:{sku}"))
                service_level = float(rng.choice([0.90, 0.92, 0.95, 0.97]))
                rows.append({
                    "sku_id": sku,
                    "holding_cost_per_unit": round(float(rng.uniform(0.5, 6.0)), 2),
                    "stockout_cost_per_unit": round(float(rng.uniform(10.0, 80.0)), 2),
                    "service_level": service_level,
                })
            return pd.DataFrame(rows)

        def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
            lower_map = {str(c).lower(): c for c in df.columns}
            for cand in candidates:
                if cand.lower() in lower_map:
                    return str(lower_map[cand.lower()])
            return None

        def _build_inventory_from_raw(
            raw_df: pd.DataFrame,
            grain_cols: list[str],
            sku_col: str | None,
            loc_col: str | None,
            extra_cols: list[str],
            combos_df: pd.DataFrame,
            inv_col_override: str | None = None,
        ) -> pd.DataFrame | None:
            # Allow caller to specify which raw column is the inventory-on-hand.
            if inv_col_override and inv_col_override in raw_df.columns:
                inv_col = inv_col_override
            else:
                inv_col = _find_column(raw_df, ["on_hand", "onhand", "inventory", "inventory_on_hand", "stock", "stock_on_hand", "qty_on_hand", "qoh"])
            if inv_col is None:
                return None

            df = raw_df.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                df = df.sort_values("date")

            # Use the latest record per grain combination as the inventory snapshot
            keys = [c for c in [sku_col, loc_col, *extra_cols] if c and c in df.columns]
            if not keys:
                return None

            latest = df.groupby(keys, as_index=False).tail(1)
            latest["on_hand"] = pd.to_numeric(latest[inv_col], errors="coerce").fillna(0.0)

            allocated_col = _find_column(df, ["allocated", "committed"])
            backorders_col = _find_column(df, ["backorders", "backorder"])
            latest["allocated"] = pd.to_numeric(latest[allocated_col], errors="coerce").fillna(0.0) if allocated_col else 0.0
            latest["backorders"] = pd.to_numeric(latest[backorders_col], errors="coerce").fillna(0.0) if backorders_col else 0.0

            # Build the same sku_id/location keys as the forecast
            latest = latest.copy()
            if sku_col is None:
                latest["sku_id"] = "ALL"
            else:
                latest["sku_id"] = latest[sku_col].astype(str)
            if loc_col is None:
                latest["location"] = "ALL"
            else:
                latest["location"] = latest[loc_col].astype(str)
            for col in extra_cols:
                if col in latest.columns:
                    latest["location"] = latest["location"] + "|" + col + "=" + latest[col].astype(str)

            inv_df = latest[["sku_id", "location", "on_hand", "allocated", "backorders"]]
            inv_df = inv_df.groupby(["sku_id", "location"], as_index=False).agg({
                "on_hand": "max",
                "allocated": "max",
                "backorders": "max",
            })

            # Ensure all forecast combinations exist (fill missing with 0)
            merged = combos_df.merge(inv_df, on=["sku_id", "location"], how="left")
            for col in ["on_hand", "allocated", "backorders"]:
                merged[col] = merged[col].fillna(0.0)
            return merged

        def _generate_inventory_df(forecast_weekly: pd.DataFrame, constraints_df: pd.DataFrame, raw_df: pd.DataFrame | None, grain_cols: list[str], sku_col: str | None, loc_col: str | None, extra_cols: list[str], inv_col_override: str | None = None) -> pd.DataFrame:
            combos_df = forecast_weekly[["sku_id", "location"]].drop_duplicates().copy()
            if raw_df is not None and isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                inv_from_raw = _build_inventory_from_raw(raw_df, grain_cols, sku_col, loc_col, extra_cols, combos_df, inv_col_override=inv_col_override)
                if inv_from_raw is not None:
                    return inv_from_raw

            # Random but sensible defaults based on average weekly demand
            mean_weekly = forecast_weekly.groupby(["sku_id", "location"])["forecast_demand"].mean().to_dict()
            rows = []
            for _, r in combos_df.iterrows():
                sku = str(r["sku_id"])
                loc = str(r["location"])
                rng = np.random.default_rng(_stable_seed(f"inv:{sku}:{loc}"))
                base = float(mean_weekly.get((sku, loc), 0.0) or 0.0)
                on_hand = max(0.0, base * rng.uniform(0.5, 3.0))
                allocated = max(0.0, on_hand * rng.uniform(0.0, 0.15))
                backorders = max(0.0, base * rng.uniform(0.0, 0.25))
                rows.append({
                    "sku_id": sku,
                    "location": loc,
                    "on_hand": int(np.ceil(on_hand)),
                    "allocated": int(np.floor(allocated)),
                    "backorders": int(np.ceil(backorders)),
                })
            return pd.DataFrame(rows)

        def _harmonize_sku_ids_to_constraints(df: pd.DataFrame, constraints: pd.DataFrame) -> pd.DataFrame:
            """
            Make df['sku_id'] match constraints['sku_id'] when PiTensor item IDs are numeric
            but constraints use a prefixed/zero-padded format like SKU_001 (or vice versa).
            """
            if "sku_id" not in df.columns or "sku_id" not in constraints.columns:
                return df

            result = df.copy()
            cons_ids = constraints["sku_id"].astype(str).str.strip()
            result["sku_id"] = result["sku_id"].astype(str).str.strip()

            # If already matches, do nothing
            if result["sku_id"].isin(cons_ids).all():
                return result

            import re

            # Find a common prefix+numeric pattern in constraints, e.g. SKU_001
            sample = next(
                (s for s in cons_ids.dropna().unique().tolist() if re.match(r"^[A-Za-z]+[_-]?\d+$", s)),
                None,
            )
            if sample is None:
                return result

            m = re.match(r"^(?P<prefix>[A-Za-z]+[_-]?)(?P<num>\d+)$", sample)
            if not m:
                return result

            prefix = m.group("prefix")
            pad = len(m.group("num"))

            # Case A: df has numeric-ish IDs (e.g. 1, 1.0) -> map to prefix+zero pad
            numeric = pd.to_numeric(result["sku_id"], errors="coerce")
            numeric_is_int = numeric.notna() & (numeric % 1 == 0)
            if numeric_is_int.all():
                result["sku_id"] = numeric.astype(int).astype(str).str.zfill(pad).radd(prefix)
                return result

            # Case B: df has prefixed IDs but constraints might be numeric (rare) -> strip prefix
            # Only apply if constraints look numeric-only.
            if cons_ids.str.fullmatch(r"\d+").all():
                stripped = result["sku_id"].str.replace(r"^[A-Za-z]+[_-]?", "", regex=True)
                if stripped.str.fullmatch(r"\d+").all():
                    result["sku_id"] = stripped
                return result

            return result

        use_latest_forecast = bool(payload.get("use_latest_forecast", True))
        auto_fill = bool(payload.get("auto_fill", True))

        combo_key = payload.get("combo_key")
        selected_sku = None
        selected_location = None
        if isinstance(combo_key, str) and "|||" in combo_key:
            parts = combo_key.split("|||", 1)
            selected_sku = parts[0]
            selected_location = parts[1]

        override_enabled = bool(payload.get("override_enabled", False))

        def _num_field(name: str) -> float | None:
            v = payload.get(name)
            if v is None:
                return None
            if isinstance(v, str) and not v.strip():
                return None
            try:
                return float(v)
            except Exception:
                return None

        raw_df = data_store.get(session_id, {}).get("df")
        inventory_df = _read_csv_optional(payload.get("inventory_csv", ""), "inventory_csv")
        # Track where inventory was sourced from for UI visibility: 'file_upload', 'file_column', 'derived', 'manual_override', 'generated'
        inventory_source = None
        if inventory_df is not None:
            inventory_source = "file_upload"
        constraints_df = _read_csv_optional(payload.get("constraints_csv", ""), "constraints_csv")
        policy_df = _read_csv_optional(payload.get("policy_csv", ""), "policy_csv")

        meta_df = None
        grain_cols: list[str] = []
        sku_col: str | None = None
        loc_col: str | None = None
        extra_cols: list[str] = []
        start_date_str: str | None = None
        horizon_months: int | None = None

        if use_latest_forecast:
            if forecast_df is None or not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
                return {"error": "No forecast data available in this session. Run a forecast first, then open Supply Planning again."}

            start_month = data_store.get(session_id, {}).get("start_month")
            months_val = data_store.get(session_id, {}).get("months")
            if not start_month or not months_val:
                return {"error": "Forecast start month / horizon not found. Please run a forecast again."}
            start_date_str = f"{start_month}-01"
            horizon_months = int(months_val)
            horizon_months = max(1, min(horizon_months, 120))

            # Map PiTensor output -> supply planning forecast schema (weekly buckets)
            if "date" not in forecast_df.columns:
                raise ValueError("Latest forecast is missing required column: date")
            if "forecast" not in forecast_df.columns:
                # Fallback to median-ish quantile if present
                quantile_fallback = next((c for c in ["forecast_p60", "forecast_p50"] if c in forecast_df.columns), None)
                if not quantile_fallback:
                    raise ValueError("Latest forecast is missing required column: forecast")
                working_df = forecast_df.rename(columns={quantile_fallback: "forecast"}).copy()
            else:
                working_df = forecast_df.copy()

            grain_cols = _infer_grain_cols_from_forecast_df(working_df)
            working_df, sku_col, loc_col, extra_cols = _build_sku_location_columns(working_df, grain_cols)

            working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")
            working_df = working_df.dropna(subset=["date"])

            # Only use forecast horizon range (ignore historical actuals)
            start_ts = pd.to_datetime(start_date_str, errors="coerce")
            if pd.isna(start_ts):
                raise ValueError(f"Invalid stored start_month: {start_month}")
            end_ts = start_ts + pd.DateOffset(months=horizon_months)
            working_df = working_df[(working_df["date"] >= start_ts) & (working_df["date"] < end_ts)]

            # Bucket monthly for time-phased planning
            working_df["period_start"] = working_df["date"].dt.to_period("M").apply(lambda p: p.start_time)
            working_df["forecast_demand"] = pd.to_numeric(working_df["forecast"], errors="coerce").fillna(0.0)

            meta_cols = [c for c in [sku_col, loc_col, *extra_cols] if c and c in working_df.columns]
            meta_df = working_df[["sku_id", "location", *meta_cols]].drop_duplicates()

            forecast_input_df = (
                working_df.groupby(["sku_id", "location", "period_start"], as_index=False)["forecast_demand"]
                .sum()
            )
        else:
            forecast_input_df = _read_csv_optional(payload.get("forecast_csv", ""), "forecast_csv")
            if forecast_input_df is None:
                raise ValueError("forecast_csv is required when use_latest_forecast=false")

            # Normalize provided forecast CSV to monthly 'period_start'
            if "period_start" not in forecast_input_df.columns:
                if "week_start" in forecast_input_df.columns:
                    forecast_input_df["week_start"] = pd.to_datetime(forecast_input_df["week_start"], errors="coerce")
                    forecast_input_df = forecast_input_df.dropna(subset=["week_start"])
                    forecast_input_df["period_start"] = forecast_input_df["week_start"].dt.to_period("M").apply(lambda p: p.start_time)
                elif "date" in forecast_input_df.columns:
                    forecast_input_df["date"] = pd.to_datetime(forecast_input_df["date"], errors="coerce")
                    forecast_input_df = forecast_input_df.dropna(subset=["date"])
                    forecast_input_df["period_start"] = forecast_input_df["date"].dt.to_period("M").apply(lambda p: p.start_time)
                else:
                    raise ValueError("forecast_csv must include one of: period_start, week_start, or date")

            if "forecast_demand" not in forecast_input_df.columns:
                raise ValueError("forecast_csv missing required column: forecast_demand")

            forecast_input_df["forecast_demand"] = pd.to_numeric(forecast_input_df["forecast_demand"], errors="coerce").fillna(0.0)
            forecast_input_df["sku_id"] = forecast_input_df["sku_id"].astype(str)
            forecast_input_df["location"] = forecast_input_df["location"].astype(str)
            forecast_input_df = forecast_input_df.groupby(["sku_id", "location", "period_start"], as_index=False)["forecast_demand"].sum()

            start_ts = pd.to_datetime(forecast_input_df["period_start"].min(), errors="coerce")
            if pd.isna(start_ts):
                raise ValueError("Could not infer start period from forecast_csv.")
            unique_months = sorted(pd.to_datetime(forecast_input_df["period_start"]).dt.to_period("M").unique())
            start_date_str = start_ts.strftime("%Y-%m-%d")
            horizon_months = max(1, min(len(unique_months), 120))

        if not auto_fill:
            if inventory_df is None:
                raise ValueError("inventory_csv is required when auto_fill=false")
            if constraints_df is None:
                raise ValueError("constraints_csv is required when auto_fill=false")
            if policy_df is None:
                raise ValueError("policy_csv is required when auto_fill=false")

        # If constraints/policy/inventory were not provided, generate defaults from forecast (and raw upload if possible).
        if constraints_df is None:
            constraints_df = _generate_constraints_df(forecast_input_df)
        if policy_df is None:
            policy_df = _generate_policy_df(constraints_df)
        if inventory_df is None:
            # Honor a user-specified inventory column if provided in the payload.
            inv_col_override = payload.get("inventory_column") if isinstance(payload.get("inventory_column"), str) else None

            # If user provided an explicit column name and raw_df exists, try to build inventory from raw using that column.
            if inv_col_override and raw_df is not None and isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                try:
                    inventory_df = _generate_inventory_df(
                        forecast_weekly=forecast_input_df.rename(columns={"period_start": "week_start"}),
                        constraints_df=constraints_df,
                        raw_df=raw_df,
                        grain_cols=grain_cols,
                        sku_col=sku_col,
                        loc_col=loc_col,
                        extra_cols=extra_cols,
                        inv_col_override=inv_col_override,
                    )
                    if inventory_df is not None:
                        inventory_source = "file_column"
                except Exception:
                    inventory_df = None

            # If still missing, prefer derived inventory from the canonical context packet if available.
            if inventory_df is None:
                try:
                    packet = build_context_packet(data_store.get(session_id, {}), combo_key=combo_key)
                    derived = packet.get("derived_inventory") if isinstance(packet, dict) else None
                    if derived and isinstance(derived, dict) and derived.get("rows"):
                        try:
                            inv_rows = pd.DataFrame(derived.get("rows"))
                            for col in ["sku_id", "location", "on_hand", "allocated", "backorders"]:
                                if col not in inv_rows.columns:
                                    inv_rows[col] = 0.0
                            inventory_df = inv_rows[["sku_id", "location", "on_hand", "allocated", "backorders"]].copy()
                            for c in ["on_hand", "allocated", "backorders"]:
                                inventory_df[c] = pd.to_numeric(inventory_df[c], errors="coerce").fillna(0.0)
                            inventory_source = "derived"
                        except Exception:
                            inventory_df = None
                    else:
                        inventory_df = None
                except Exception:
                    inventory_df = None

            # Final fallback: generate defaults from forecast and any raw upload.
            if inventory_df is None:
                inventory_df = _generate_inventory_df(
                    forecast_weekly=forecast_input_df.rename(columns={"period_start": "week_start"}),
                    constraints_df=constraints_df,
                    raw_df=raw_df,
                    grain_cols=grain_cols,
                    sku_col=sku_col,
                    loc_col=loc_col,
                    extra_cols=extra_cols,
                    inv_col_override=inv_col_override,
                )
                inventory_source = inventory_source or "generated"

        # Harmonize sku_id formats (e.g. item=1 -> SKU_001) so constraints/policy match the forecast.
        forecast_input_df = _harmonize_sku_ids_to_constraints(forecast_input_df, constraints_df)
        inventory_df = _harmonize_sku_ids_to_constraints(inventory_df, constraints_df)
        policy_df = _harmonize_sku_ids_to_constraints(policy_df, constraints_df)
        if meta_df is not None:
            meta_df = _harmonize_sku_ids_to_constraints(meta_df, constraints_df)

        if selected_sku is not None:
            sel_df = pd.DataFrame({"sku_id": [selected_sku]})
            sel_df = _harmonize_sku_ids_to_constraints(sel_df, constraints_df)
            selected_sku = str(sel_df.iloc[0]["sku_id"])

        # Apply user overrides for selected series (constraints, policy, inventory, and/or monthly demand).
        if override_enabled and selected_sku is not None and selected_location is not None:
            # Upsert constraints
            for col in ["sku_id", "supplier", "lead_time_days", "moq", "order_multiple", "max_capacity_per_week", "shelf_life_days"]:
                if col not in constraints_df.columns:
                    constraints_df[col] = np.nan
            mask_c = constraints_df["sku_id"].astype(str) == str(selected_sku)
            if not mask_c.any():
                constraints_df = pd.concat([constraints_df, pd.DataFrame([{"sku_id": str(selected_sku), "supplier": "SUP_OVERRIDE"}])], ignore_index=True)
                mask_c = constraints_df["sku_id"].astype(str) == str(selected_sku)

            lead_time_days = _num_field("override_lead_time_days")
            moq = _num_field("override_moq")
            order_multiple = _num_field("override_order_multiple")
            max_cap_week = _num_field("override_max_capacity_per_week")
            if lead_time_days is not None:
                constraints_df.loc[mask_c, "lead_time_days"] = lead_time_days
            if moq is not None:
                constraints_df.loc[mask_c, "moq"] = moq
            if order_multiple is not None and order_multiple > 0:
                constraints_df.loc[mask_c, "order_multiple"] = order_multiple
            if max_cap_week is not None and max_cap_week > 0:
                constraints_df.loc[mask_c, "max_capacity_per_week"] = max_cap_week

            # Upsert policy
            for col in ["sku_id", "service_level", "holding_cost_per_unit", "stockout_cost_per_unit"]:
                if col not in policy_df.columns:
                    policy_df[col] = np.nan
            mask_p = policy_df["sku_id"].astype(str) == str(selected_sku)
            if not mask_p.any():
                policy_df = pd.concat([policy_df, pd.DataFrame([{"sku_id": str(selected_sku)}])], ignore_index=True)
                mask_p = policy_df["sku_id"].astype(str) == str(selected_sku)
            service_level = _num_field("override_service_level")
            if service_level is not None and 0 < service_level < 1:
                policy_df.loc[mask_p, "service_level"] = service_level

            # Upsert inventory snapshot (skip if user explicitly supplied an inventory column)
            inv_col_override = payload.get("inventory_column") if isinstance(payload.get("inventory_column"), str) and payload.get("inventory_column") else None
            # If the user selected a raw inventory column, prefer that and do not apply manual inventory overrides here.
            if not inv_col_override:
                # Upsert inventory snapshot
                for col in ["sku_id", "location", "on_hand", "allocated", "backorders"]:
                    if col not in inventory_df.columns:
                        inventory_df[col] = 0.0
                mask_i = (inventory_df["sku_id"].astype(str) == str(selected_sku)) & (inventory_df["location"].astype(str) == str(selected_location))
                if not mask_i.any():
                    inventory_df = pd.concat([inventory_df, pd.DataFrame([{"sku_id": str(selected_sku), "location": str(selected_location)}])], ignore_index=True)
                    mask_i = (inventory_df["sku_id"].astype(str) == str(selected_sku)) & (inventory_df["location"].astype(str) == str(selected_location))

                on_hand = _num_field("override_on_hand")
                allocated = _num_field("override_allocated")
                backorders = _num_field("override_backorders")
                any_override_applied = False
                if on_hand is not None and on_hand >= 0:
                    inventory_df.loc[mask_i, "on_hand"] = on_hand
                    any_override_applied = True
                if allocated is not None and allocated >= 0:
                    inventory_df.loc[mask_i, "allocated"] = allocated
                    any_override_applied = True
                if backorders is not None and backorders >= 0:
                    inventory_df.loc[mask_i, "backorders"] = backorders
                    any_override_applied = True
                if any_override_applied:
                    inventory_source = "manual_override"

            # Monthly forecast override for selected series
            override_forecast_csv = payload.get("override_forecast_csv")
            if isinstance(override_forecast_csv, str) and override_forecast_csv.strip():
                f_override = pd.read_csv(StringIO(override_forecast_csv.strip()))
                if "period_start" in f_override.columns and "forecast_demand" in f_override.columns:
                    f_override["period_start"] = pd.to_datetime(f_override["period_start"], errors="coerce")
                    f_override["forecast_demand"] = pd.to_numeric(f_override["forecast_demand"], errors="coerce").fillna(0.0)
                    f_override = f_override.dropna(subset=["period_start"])
                    f_override = f_override[(f_override["period_start"] >= start_ts) & (f_override["period_start"] < end_ts)]
                    f_override["period_start"] = f_override["period_start"].dt.to_period("M").apply(lambda p: p.start_time)
                    f_override = f_override.groupby(["period_start"], as_index=False)["forecast_demand"].sum()

                    # Replace series rows for the whole horizon (ensures all months exist)
                    base_series = pd.DataFrame({"period_start": pd.date_range(start=start_ts, periods=horizon_months or 10, freq="MS")})
                    base_series = base_series.merge(f_override, on="period_start", how="left")
                    base_series["forecast_demand"] = base_series["forecast_demand"].fillna(0.0)
                    base_series["sku_id"] = str(selected_sku)
                    base_series["location"] = str(selected_location)

                    forecast_input_df = forecast_input_df[~((forecast_input_df["sku_id"].astype(str) == str(selected_sku)) & (forecast_input_df["location"].astype(str) == str(selected_location)))]
                    forecast_input_df = pd.concat([forecast_input_df, base_series[["sku_id", "location", "period_start", "forecast_demand"]]], ignore_index=True)

        supply_plan = generate_time_phased_supply_plan(
            forecast_df=forecast_input_df,
            inventory_df=inventory_df,
            constraints_df=constraints_df,
            policy_df=policy_df,
            start_date=start_date_str or datetime.now().strftime("%Y-%m-%d"),
            months=horizon_months or 10,
            strict=False,
        )

        if meta_df is not None and not meta_df.empty:
            supply_plan = supply_plan.merge(meta_df, on=["sku_id", "location"], how="left")
            # Prefer showing original grain columns first (item/store/etc), then the supply recommendation columns.
            preferred_front = [c for c in [sku_col, loc_col, *extra_cols] if c and c in supply_plan.columns]
            remaining = [c for c in supply_plan.columns if c not in preferred_front]
            supply_plan = supply_plan[preferred_front + remaining]

        # Store a user-facing plan for download (drop redundant sku_id/location when grain columns exist).
        supply_plan_export = supply_plan.copy()
        if meta_df is not None and not meta_df.empty:
            if "sku_id" in supply_plan_export.columns and sku_col and sku_col in supply_plan_export.columns:
                supply_plan_export = supply_plan_export.drop(columns=["sku_id"])
            if "location" in supply_plan_export.columns and loc_col and loc_col in supply_plan_export.columns:
                supply_plan_export = supply_plan_export.drop(columns=["location"])

        data_store[session_id]["supply_plan_df"] = supply_plan_export
        # Keep a full copy for downstream analysis (risks/actions/assistant context).
        data_store[session_id]["supply_plan_full_df"] = supply_plan.copy()

        # Persist supply plan history (best-effort).
        if user_email:
            try:
                import history_store
                run_id = data_store.get(session_id, {}).get("forecast_run_id")
                if run_id:
                    plan_params = {
                        "combo_key": payload.get("combo_key"),
                        "start_date": payload.get("start_date"),
                        "months": payload.get("months"),
                        "override_enabled": bool(payload.get("override_enabled") or False),
                        "inventory_column": payload.get("inventory_column"),
                    }
                    history_store.save_supply_plan(
                        user_email,
                        int(run_id),
                        params=plan_params,
                        supply_export_df=supply_plan_export,
                        supply_full_df=supply_plan,
                    )
                    data_store[session_id]["history_last_error"] = None
            except Exception as e:
                logging.warning(f"[HISTORY] Failed to save supply plan for {user_email}: {e}")
                try:
                    data_store[session_id]["history_last_error"] = f"Failed to save supply plan history: {e}"
                except Exception:
                    pass

        # Filter plan to selected series for display/plot
        plot_df = supply_plan.copy()
        if selected_sku is not None and selected_location is not None:
            plot_df = plot_df[(plot_df["sku_id"].astype(str) == str(selected_sku)) & (plot_df["location"].astype(str) == str(selected_location))]
        else:
            first = plot_df[["sku_id", "location"]].drop_duplicates().head(1)
            if not first.empty:
                selected_sku = str(first.iloc[0]["sku_id"])
                selected_location = str(first.iloc[0]["location"])
                plot_df = plot_df[(plot_df["sku_id"].astype(str) == selected_sku) & (plot_df["location"].astype(str) == selected_location)]

        plot_df = plot_df.sort_values("period_start").head(horizon_months or 10)

        plot_df_display = plot_df.copy()
        if meta_df is not None and not meta_df.empty:
            if "sku_id" in plot_df_display.columns and sku_col and sku_col in plot_df_display.columns:
                plot_df_display = plot_df_display.drop(columns=["sku_id"])
            if "location" in plot_df_display.columns and loc_col and loc_col in plot_df_display.columns:
                plot_df_display = plot_df_display.drop(columns=["location"])

        head = plot_df_display[[c for c in plot_df_display.columns if c != "explanation"]].head(horizon_months or 10).to_string(index=False)

        # Sawtooth chart: inventory over time with reorder point + safety stock
        import plotly.graph_objects as go
        from plotly.utils import PlotlyJSONEncoder

        points_x = []
        points_y = []
        hover = []
        prev_end = None
        for _, row in plot_df.iterrows():
            m_start = pd.to_datetime(row["period_start"])
            m_end = (m_start + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
            begin_inv = float(row.get("beginning_on_hand", 0.0) or 0.0)
            end_inv = float(row.get("ending_on_hand", 0.0) or 0.0)

            if prev_end is None:
                points_x.append(m_start)
                points_y.append(begin_inv)
                hover.append(f"Month start<br>Begin inv: {begin_inv:.0f}")
            else:
                # vertical jump at same x
                points_x.append(m_start)
                points_y.append(prev_end)
                hover.append(f"Month start<br>Prev end inv: {prev_end:.0f}")
                points_x.append(m_start)
                points_y.append(begin_inv)
                hover.append(f"Month start<br>Begin inv: {begin_inv:.0f}")

            points_x.append(m_end)
            points_y.append(end_inv)
            hover.append(
                f"{m_start.strftime('%b %Y')}<br>"
                f"Demand: {float(row.get('forecast_demand', 0.0) or 0.0):.0f}<br>"
                f"Receipts: {float(row.get('receipts', 0.0) or 0.0):.0f}<br>"
                f"Order: {float(row.get('order_qty', 0.0) or 0.0):.0f}<br>"
                f"End inv: {end_inv:.0f}"
            )
            prev_end = end_inv

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=points_x,
            y=points_y,
            mode="lines+markers",
            name="Projected On-hand",
            line=dict(color="#2b6ef2", width=3),
            marker=dict(size=7),
            hovertext=hover,
            hoverinfo="text",
        ))

        def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
            ts = pd.to_datetime(ts)
            return (ts + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)

        def _step_xy(df: pd.DataFrame, col: str) -> tuple[list[pd.Timestamp], list[float]]:
            xs: list[pd.Timestamp] = []
            ys: list[float] = []
            for _, r in df.iterrows():
                m_start = pd.to_datetime(r["period_start"])
                m_end = _month_end(m_start)
                v = pd.to_numeric(r.get(col), errors="coerce")
                v = float(v) if pd.notna(v) else float("nan")
                xs.extend([m_start, m_end])
                ys.extend([v, v])
            return xs, ys

        rp_x, rp_y = _step_xy(plot_df, "reorder_point")
        ss_x, ss_y = _step_xy(plot_df, "safety_stock")
        tl_x, tl_y = _step_xy(plot_df, "target_level")

        fig.add_trace(go.Scatter(
            x=rp_x,
            y=rp_y,
            mode="lines",
            name="Reorder Point",
            line=dict(color="#d34a4a", width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=ss_x,
            y=ss_y,
            mode="lines",
            name="Safety Stock",
            line=dict(color="#6c757d", width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=tl_x,
            y=tl_y,
            mode="lines",
            name="Order-up-to Target",
            line=dict(color="#7b2cbf", width=2, dash="dash"),
        ))
        fig.update_layout(
            title="Projected Inventory (Sawtooth)",
            xaxis_title="Month",
            yaxis_title="Units",
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=30, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        fig_json = json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder)

        # Build a JSON-serializable table for the UI.
        table_df = plot_df_display[[c for c in plot_df_display.columns if c != "explanation"]].head(min(int(horizon_months or 10), 36)).copy()
        table_columns = [str(c) for c in table_df.columns]

        def _json_safe(v: object) -> object:
            try:
                if v is None:
                    return None
                if isinstance(v, pd.Timestamp):
                    return v.strftime("%Y-%m-%d")
                if isinstance(v, datetime):
                    return v.strftime("%Y-%m-%d")
                if isinstance(v, float):
                    if not pd.notna(v):
                        return None
                    return float(v)
                if isinstance(v, int):
                    return int(v)
                # numpy types
                if "numpy" in type(v).__module__:
                    if pd.isna(v):
                        return None
                    try:
                        return v.item()
                    except Exception:
                        return str(v)
                if pd.isna(v):  # type: ignore[arg-type]
                    return None
                return v
            except Exception:
                return str(v)

        table_rows = []
        for _, r in table_df.iterrows():
            row = {str(k): _json_safe(v) for k, v in r.to_dict().items()}
            table_rows.append(row)

        return {"head": head, "plot_json": fig_json, "table_columns": table_columns, "table_rows": table_rows, "inventory_source": inventory_source}
    except Exception as e:
        logging.exception(f"[LOG] Error in supply planning: {e}")
        return {"error": f"Supply planning error: {e}"}

# Download endpoint for supply plan
@app.get("/download_supply_plan")
async def download_supply_plan(request: Request):
    session_id = _session_id_from_request(request)
    supply_plan = data_store.get(session_id, {}).get("supply_plan_df")
    if supply_plan is None or supply_plan.empty:
        raise HTTPException(status_code=404, detail="No supply plan available. Please generate it first.")
    stream = io.StringIO()
    supply_plan.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=order_recommendations.csv"})

# --- Planning Pipeline Endpoints ---
@app.get("/planning/context")
async def planning_context(request: Request, combo_key: Optional[str] = None):
    session_id = _session_id_from_request(request)
    session = data_store.get(session_id, {})
    return build_planning_context(session, combo_key=combo_key)

@app.post("/risks/detect")
async def risks_detect(request: Request, payload: Dict = Body(default={})):
    session_id = _session_id_from_request(request)
    combo_key = payload.get("combo_key")
    session = data_store.get(session_id, {})
    context = build_planning_context(session, combo_key=combo_key)
    risks = detect_risks(session, context=context)
    return {"context": context, "risks": risks}

@app.post("/actions/generate")
async def actions_generate(request: Request, payload: Dict = Body(default={})):
    session_id = _session_id_from_request(request)
    combo_key = payload.get("combo_key")
    session = data_store.get(session_id, {})
    context = payload.get("context") or build_planning_context(session, combo_key=combo_key)
    risks = payload.get("risks") or detect_risks(session, context=context)
    actions = generate_actions(session, context=context, risks=risks)
    return {"context": context, "risks": risks, "actions": actions}

@app.post("/ai/recommendations")
async def ai_recommendations(request: Request, payload: Dict = Body(default={})):
    """
    Uses the configured LLM provider (Gemini/OpenAI) to convert context+risks+actions into a human-friendly plan.
    Falls back to the local assistant when the LLM is unavailable.
    """
    session_id = _session_id_from_request(request)
    combo_key = payload.get("combo_key")
    question = payload.get("question") or "Provide prioritized recommendations and next steps."
    request_id = payload.get("request_id")
    session = data_store.get(session_id, {})
    # Planning recommendations are intended to use OpenAI when configured.
    # If OPENAI_API_KEY isn't set, fall back deterministically to local (no external calls).
    openai_configured = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    attempted_provider = "openai"

    context = payload.get("context") or build_planning_context(session, combo_key=combo_key)
    risks = payload.get("risks") or detect_risks(session, context=context)
    actions = payload.get("actions") or generate_actions(session, context=context, risks=risks)

    # Canonical packet: Raw Sales + Forecast Output + Supply Plan & Risks.
    context_packet = _build_llm_context_packet(session, combo_key=combo_key)

    cache_key = None
    if openai_configured and isinstance(request_id, str) and request_id.strip():
        cache_key = f"recs:{attempted_provider}:{session_id}:{request_id.strip()}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"context": context, "risks": risks, "actions": actions, "answer": cached, "provider": attempted_provider, "llm_provider": attempted_provider, "llm_error": None, "cached": True}

    # De-dupe identical recommendation questions briefly (per session).
    rec_hash = hashlib.sha256(f"{combo_key}|{question}".encode("utf-8")).hexdigest()[:16]
    hash_key = f"recsq:{attempted_provider}:{session_id}:{rec_hash}"
    if openai_configured:
        cached = _cache_get(hash_key)
        if cached is not None:
            return {"context": context, "risks": risks, "actions": actions, "answer": cached, "provider": attempted_provider, "llm_provider": attempted_provider, "llm_error": None, "cached": True}

    lock = _get_lock(session_id)
    async with lock:
        # Re-check cache after waiting for in-flight requests.
        if cache_key and openai_configured:
            cached = _cache_get(cache_key)
            if cached is not None:
                return {"context": context, "risks": risks, "actions": actions, "answer": cached, "provider": attempted_provider, "llm_provider": attempted_provider, "llm_error": None, "cached": True}
        if openai_configured:
            cached = _cache_get(hash_key)
            if cached is not None:
                return {"context": context, "risks": risks, "actions": actions, "answer": cached, "provider": attempted_provider, "llm_provider": attempted_provider, "llm_error": None, "cached": True}

        try:
            try:
                if not openai_configured:
                    raise ValueError("OPENAI_API_KEY is not set.")
                answer = await answer_question_openai(question, context_packet=context_packet)
                provider_used = attempted_provider
                llm_provider = attempted_provider
                llm_error = None
            except Exception as e:
                # Keep this pipeline deterministic/safe: fall back to local (no extra LLM calls).
                provider_used = "local"
                llm_provider = attempted_provider
                llm_error = _llm_unavailable_prefix(attempted_provider, e)
                answer = _local_chat_answer(question, session)
            # Cache only successful OpenAI answers to keep provider labels accurate.
            if provider_used == attempted_provider:
                if cache_key:
                    _cache_set(cache_key, answer)
                _cache_set(hash_key, answer)
            return {
                "context": context,
                "risks": risks,
                "actions": actions,
                "answer": answer,
                "provider": provider_used,
                "llm_provider": llm_provider,
                "llm_error": llm_error,
                "cached": False,
            }
        except Exception as e:
            local = _local_chat_answer(question, session)
            msg = _llm_unavailable_prefix(attempted_provider, e)
            _assistant_debug(f"ai_recommendations fallback to local: {msg}")
            return {
                "context": context,
                "risks": risks,
                "actions": actions,
                "answer": local,
                "provider": "local",
                "llm_provider": attempted_provider,
                "llm_error": msg,
                "cached": False,
            }

@app.post("/planning/full")
async def planning_full(request: Request, payload: Dict = Body(default={})):
    """
    Convenience endpoint that runs the whole pipeline:
      /planning/context -> /risks/detect -> /actions/generate -> /ai/recommendations
    """
    session_id = _session_id_from_request(request)
    combo_key = payload.get("combo_key")
    request_id = payload.get("request_id")
    session = data_store.get(session_id, {})

    try:
        context = build_planning_context(session, combo_key=combo_key)
        risks = detect_risks(session, context=context)
        actions = generate_actions(session, context=context, risks=risks)

        recs_payload = {
            "combo_key": combo_key,
            "context": context,
            "risks": risks,
            "actions": actions,
            "question": payload.get("question"),
            "request_id": request_id,
        }
        recs = await ai_recommendations(request, recs_payload)
        try:
            # Store last planning state for follow-up Q&A (per session).
            store = data_store.setdefault(session_id, session)
            context_packet = _build_llm_context_packet(session, combo_key=combo_key)
            store["planning_last"] = {
                "ts": time.time(),
                "combo_key": combo_key,
                "question": payload.get("question"),
                "context": context,
                "risks": risks,
                "actions": actions,
                "recommendations": recs.get("answer") if isinstance(recs, dict) else None,
                "provider": recs.get("provider") if isinstance(recs, dict) else None,
                "llm_provider": recs.get("llm_provider") if isinstance(recs, dict) else None,
                "llm_error": recs.get("llm_error") if isinstance(recs, dict) else None,
                "context_packet": context_packet,
            }
        except Exception:
            pass
        # Do not cache entire dict in _assistant_cache; it is shared with chat and intended for short strings.
        return recs
    except Exception as e:
        # Return JSON error so the frontend can parse the response body instead of receiving HTML.
        import traceback, pathlib
        tb = traceback.format_exc()
        _assistant_debug(f"/planning/full error: {e}\n{tb}")
        try:
            p = pathlib.Path("planning_error.log").resolve()
            with p.open("a", encoding="utf-8") as fh:
                fh.write(f"--- /planning/full error at {time.asctime()} ---\n")
                fh.write(tb)
                fh.write("\n\n")
        except Exception:
            # best-effort logging; ignore failures to avoid masking original error
            pass
        return JSONResponse({"detail": "Internal server error (see planning_error.log on server)."}, status_code=500)

@app.post("/planning/followup")
async def planning_followup(request: Request, payload: Dict = Body(default={})):
    """
    Follow-up Q&A about the last generated planning insights (risks/actions/recommendations).
    The client can send short conversation history so users can ask subsequent questions.
    """
    session_id = _session_id_from_request(request)
    question = (payload.get("question") or "").strip()
    request_id = payload.get("request_id")
    history = payload.get("history") or []
    combo_key = payload.get("combo_key")

    if not question:
        return {"answer": "Please enter a question.", "provider": "local", "llm_provider": _default_provider(), "llm_error": None, "cached": False}

    session = data_store.get(session_id, {})
    last = session.get("planning_last") if isinstance(session, dict) else None
    if not isinstance(last, dict) or not last.get("context"):
        raise HTTPException(status_code=400, detail="No planning insights found. Click Generate first.")

    if combo_key and last.get("combo_key") and str(combo_key) != str(last.get("combo_key")):
        raise HTTPException(status_code=400, detail="Planning insights do not match the currently selected series. Click Generate again.")

    openai_configured = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    attempted_provider = "openai"
    cache_key = None
    if openai_configured and isinstance(request_id, str) and request_id.strip():
        cache_key = f"planqa:{attempted_provider}:{session_id}:{request_id.strip()}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"answer": cached, "provider": attempted_provider, "llm_provider": attempted_provider, "llm_error": None, "cached": True}

    # Keep history short to avoid blowing up prompt size.
    hist_lines: list[str] = []
    try:
        if isinstance(history, list):
            for m in history[-10:]:
                if not isinstance(m, dict):
                    continue
                role = (m.get("role") or "").strip().lower()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role not in ("user", "assistant"):
                    continue
                prefix = "User" if role == "user" else "Assistant"
                hist_lines.append(f"{prefix}: {content}")
    except Exception:
        hist_lines = []

    history_text = "\n".join(hist_lines).strip()
    user_message = question
    if history_text:
        user_message = f"{question}\n\nConversation so far:\n{history_text}"

    # Use canonical data blocks from the last run (or rebuild) and answer using a strict contract.
    context_packet = last.get("context_packet") if isinstance(last, dict) else None
    if not isinstance(context_packet, dict):
        context_packet = _build_llm_context_packet(session, combo_key=combo_key)

    lock = _get_lock(session_id)
    async with lock:
        if cache_key and openai_configured:
            cached = _cache_get(cache_key)
            if cached is not None:
                return {"answer": cached, "provider": attempted_provider, "llm_provider": attempted_provider, "llm_error": None, "cached": True}

        try:
            try:
                if not openai_configured:
                    raise ValueError("OPENAI_API_KEY is not set.")
                answer = await answer_question_openai(user_message, context_packet=context_packet)
                provider_used = attempted_provider
                llm_provider = attempted_provider
                llm_error = None
            except Exception as e:
                provider_used = "local"
                llm_provider = attempted_provider
                llm_error = _llm_unavailable_prefix(attempted_provider, e)
                answer = _local_chat_answer(user_message, session)

            if cache_key and provider_used == attempted_provider:
                _cache_set(cache_key, answer)
            return {"answer": answer, "provider": provider_used, "llm_provider": llm_provider, "llm_error": llm_error, "cached": False}
        except Exception as e:
            msg = _llm_unavailable_prefix(attempted_provider, e)
            try:
                local = _local_chat_answer(user_message, session)
            except Exception:
                local = "LLM unavailable; please retry after fixing the LLM connection."
            return {"answer": local, "provider": "local", "llm_provider": attempted_provider, "llm_error": msg, "cached": False}

# --- Assistant Chat Endpoint ---
@app.post("/chat")
async def chat_endpoint(request: Request, payload: Dict = Body(...)):
    """
     Receives a chat message from the frontend and returns an AI-generated answer
     (Gemini/OpenAI) using the uploaded raw data, the generated forecast, and the generated supply plan as context.
    """
    user_message = payload.get("message", "")
    if not user_message.strip():
        return {"answer": "Please enter a question about your forecast results."}

    session_id = _session_id_from_request(request)
    session = data_store.get(session_id, {})
    combo_key = payload.get("combo_key")
    request_id = payload.get("request_id")
    provider = _default_provider()
    _assistant_debug(f"/chat start: session={session_id} request_id={request_id!r}")
    cache_key = None
    if isinstance(request_id, str) and request_id.strip():
        cache_key = f"chat:{provider}:{session_id}:{request_id.strip()}"
        cached = _cache_get(cache_key)
        if cached is not None:
            _assistant_debug(f"/chat cache hit (request_id): {cache_key}")
            return {"answer": cached, "provider": provider, "llm_provider": provider, "llm_error": None, "cached": True}

    # Also de-dupe identical questions briefly (helps with browser/proxy retries).
    q_hash = hashlib.sha256(user_message.strip().encode("utf-8")).hexdigest()[:16]
    hash_key = f"chatq:{provider}:{session_id}:{q_hash}"
    cached = _cache_get(hash_key)
    if cached is not None:
        _assistant_debug(f"/chat cache hit (question_hash): {hash_key}")
        return {"answer": cached, "provider": provider, "llm_provider": provider, "llm_error": None, "cached": True}

    lock = _get_lock(session_id)
    async with lock:
        # Re-check cache after waiting for another in-flight request.
        if cache_key:
                cached = _cache_get(cache_key)
                if cached is not None:
                    _assistant_debug(f"/chat cache hit after lock (request_id): {cache_key}")
                    return {"answer": cached, "provider": provider, "llm_provider": provider, "llm_error": None, "cached": True}
        cached = _cache_get(hash_key)
        if cached is not None:
            _assistant_debug(f"/chat cache hit after lock (question_hash): {hash_key}")
            return {"answer": cached, "provider": provider, "llm_provider": provider, "llm_error": None, "cached": True}

        try:
            _assistant_debug(f"/chat calling provider={provider}")

            # Debug: Check session data
            logging.info(f"[SESSION DEBUG] Session has forecast_df: {session.get('forecast_df') is not None}")
            logging.info(f"[SESSION DEBUG] Session keys: {list(session.keys())}")

            context_packet = _build_llm_context_packet(session, combo_key=combo_key)
            logging.info(f"[PACKET DEBUG] Context packet keys: {list(context_packet.keys())}")

            # Ensure RAG system has indexed the data
            if not retriever.get_retriever().initialized:
                try:
                    logging.info(f"[RAG DEBUG] Indexing session data...")
                    await _index_session_data(session)
                    logging.info(f"[RAG DEBUG] Indexing complete")
                except Exception as e:
                    logging.warning(f"Failed to index data for RAG: {e}")

            try:
                answer = await answer_question_openai(user_message, context_packet=context_packet)
                provider_used = "openai"
                llm_provider = "openai"
                llm_error = None
            except Exception as e:
                provider_used = "local"
                llm_provider = "openai"
                llm_error = _llm_unavailable_prefix("openai", e)
                answer = _local_chat_answer(user_message, session)
            if cache_key:
                _cache_set(cache_key, answer)
            _cache_set(hash_key, answer)
            _assistant_debug("/chat done")
            return {"answer": answer, "provider": provider_used, "llm_provider": llm_provider, "llm_error": llm_error, "cached": False}
        except Exception as e:
            # Fallback to local rules if LLM is not configured / unreachable.
            try:
                local = _local_chat_answer(user_message, session)
                msg = _llm_unavailable_prefix(provider, e)
                _assistant_debug(f"/chat fallback to local: {msg}")
                return {"answer": local, "provider": "local", "llm_provider": provider, "llm_error": msg, "cached": False}
            except Exception:
                return {"answer": f"Assistant error: {e}", "provider": "local", "cached": False}

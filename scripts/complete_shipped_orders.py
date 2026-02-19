#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
complete_shipped_orders.py

Rules (status_id=2 only):
1) If shipping method is NOT "Ship By Weight" => move to status_id=10 (Completed) immediately.
2) If shipping method IS "Ship By Weight":
   - Pull all tracking numbers from the order (multiple shipments supported).
   - If there are NO tracking numbers: ALERT (but do NOT fail the run).
   - If ANY tracking numbers are obviously invalid (e.g., "1"): ALERT (but do NOT fail the run).
   - Use FedEx Track API to check ALL tracking numbers (for trackable ones).
   - Only move the BigCommerce order to Completed when:
       * there are NO missing tracking numbers,
       * there are NO invalid tracking numbers,
       * and ALL tracking numbers are delivered.

Alerting:
- Missing tracking: write markdown (ALERT_PATH) + set step output missing_tracking=true
- Invalid tracking: write markdown (INVALID_TRACKING_ALERT_PATH) + set step output invalid_tracking=true
- FedEx empty/unparseable results (optional): write markdown (FEDEX_EMPTY_ALERT_PATH) + output fedex_empty=true

Caching:
- Each run lists shipped orders.
- Shipping method + tracking numbers are fetched once per order then cached.
- Cache prunes itself when an order is no longer in Shipped status.

Notes:
- Tracking format validation is heuristic. FedEx tracking numbers are commonly 12 or 15 digits, sometimes 20 or 22.
  This script flags obviously-bad values (too short, illegal characters), and still allows “unknown-but-plausible”
  values through to the FedEx API.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

BC_BASE_URL = "https://api.bigcommerce.com/stores"

FEDEX_BASE_URLS = {
    "sandbox": "https://apis-sandbox.fedex.com",
    "prod": "https://apis.fedex.com",
}


# ----------------------------
# Helpers
# ----------------------------
def _get_env(name: str, required: bool = False, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if required and (val is None or str(val).strip() == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return "" if val is None else str(val)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_mkdir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def retry_request(func, retries: int = 8, delay: float = 2.0, backoff: float = 1.5):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except requests.RequestException as e:
            last_err = e
            logging.warning(f"Request failed (attempt {attempt}/{retries}): {e}")
            time.sleep(delay)
            delay *= backoff
    raise RuntimeError(f"Max retries reached for an API call. Last error: {last_err}")


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _write_step_output(key: str, value: str) -> None:
    """
    Write to GitHub Actions step outputs if GITHUB_OUTPUT is available.
    """
    out_path = os.getenv("GITHUB_OUTPUT")
    if not out_path:
        return
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
    except Exception:
        # Don't break the job over output writing
        pass


def _parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


# ----------------------------
# Tracking validation (heuristic)
# ----------------------------
_ALLOWED_TRACKING_CHARS = re.compile(r"^[A-Za-z0-9]+$")  # after stripping spaces/hyphens
_DIGITS_ONLY = re.compile(r"^[0-9]+$")


def classify_tracking_number(raw: str) -> Tuple[bool, str]:
    """
    Returns (is_obviously_invalid, reason).
    - Flags obviously bad values so they trigger alerts even if FedEx returns empty output.
    - Allows unknown-but-plausible values to still go to FedEx.
    """
    if raw is None:
        return True, "missing"

    t = str(raw).strip()
    if not t:
        return True, "missing"

    # remove common separators
    compact = t.replace(" ", "").replace("-", "")
    if not compact:
        return True, "missing"

    # illegal characters?
    if not _ALLOWED_TRACKING_CHARS.match(compact):
        return True, "illegal_characters"

    # too short to be a real carrier tracking number (catches "1", "123", etc.)
    if len(compact) < 8:
        return True, "too_short"

    if _DIGITS_ONLY.match(compact):
        if len(compact) in {12, 15, 20, 22}:
            return False, "looks_like_fedex_common_length"
        return False, "digits_unknown_length"

    return False, "alphanumeric_unknown_format"


# ----------------------------
# BigCommerce
# ----------------------------
def bc_headers(auth_token: str) -> Dict[str, str]:
    return {
        "X-Auth-Token": auth_token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def bc_get_json(auth_token: str, url: str, params: dict | None = None, *, timeout: int = 30, max_retries: int = 4):
    """
    GET JSON with retries and strong diagnostics.
    """
    headers = bc_headers(auth_token)

    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)

        status = resp.status_code
        ctype = (resp.headers.get("Content-Type") or "").lower()
        text = resp.text or ""
        snippet = text[:500].replace("\n", " ").replace("\r", " ")

        # Retryable statuses
        if status in (429, 500, 502, 503, 504):
            logging.warning(
                "BigCommerce %s for %s (attempt %d/%d). Body snippet: %s",
                status, url, attempt, max_retries, snippet
            )
            time.sleep(backoff)
            backoff *= 2
            continue

        # Hard failures with clear messaging
        if status in (401, 403):
            raise RuntimeError(
                f"BigCommerce auth error {status} calling {url}. "
                f"Check BC_AUTH_TOKEN (and store scope). Body snippet: {snippet}"
            )
        if status == 404:
            raise RuntimeError(
                f"BigCommerce 404 calling {url}. "
                f"Often means BC_STORE_ID is wrong/empty or endpoint path is wrong. Body snippet: {snippet}"
            )

        # Any other non-2xx
        if not (200 <= status < 300):
            raise RuntimeError(
                f"BigCommerce HTTP {status} calling {url}. Body snippet: {snippet}"
            )

        # JSON decode with better diagnostics
        try:
            return resp.json()
        except Exception:
            raise RuntimeError(
                f"BigCommerce returned non-JSON for {url} (HTTP {status}, Content-Type={ctype}). "
                f"Body snippet: {snippet}"
            )

    raise RuntimeError(f"BigCommerce failed after {max_retries} retries for {url}.")


def bc_put_json(auth_token: str, url: str, payload: Dict[str, Any], timeout: int = 30) -> Any:
    def _do():
        return requests.put(url, headers=bc_headers(auth_token), json=payload, timeout=timeout)

    resp = retry_request(_do)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"BigCommerce PUT failed {resp.status_code} {url}: {resp.text[:500]}", response=resp)

    try:
        return resp.json()
    except Exception:
        return {"status_code": resp.status_code, "text": resp.text}


def list_orders_by_status(auth_token: str, store_id: str, status_id: int, limit: int = 250) -> List[Dict[str, Any]]:
    orders: List[Dict[str, Any]] = []
    page = 1

    while True:
        url = f"{BC_BASE_URL}/{store_id}/v2/orders"
        batch = bc_get_json(auth_token, url, params={"status_id": status_id, "page": page, "limit": limit})
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected BigCommerce orders response (expected list). Got: {type(batch)}")
        if not batch:
            break
        orders.extend(batch)
        if len(batch) < limit:
            break
        page += 1

    return orders


def fetch_shipping_method(auth_token: str, store_id: str, order_id: int) -> str:
    url = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipping_addresses"
    addrs = bc_get_json(auth_token, url)
    if isinstance(addrs, list) and addrs:
        return (addrs[0].get("shipping_method", "") or "").strip()
    return ""


def fetch_tracking_numbers_and_provider(auth_token: str, store_id: str, order_id: int) -> Tuple[List[str], str]:
    """
    Pull tracking numbers from shipments endpoint; if empty, fall back to shipping_addresses.
    Dedupes and preserves order.
    """
    tracking_numbers: List[str] = []
    provider = ""

    url_ship = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipments"
    shipments = bc_get_json(auth_token, url_ship)

    if isinstance(shipments, list):
        for sh in shipments:
            tn = (sh.get("tracking_number", "") or "").strip()
            # IMPORTANT: keep even weird values like "1" so we can alert on them.
            if tn != "":
                tracking_numbers.append(tn)

            if not provider:
                provider = (sh.get("shipping_provider", "") or "").strip()

    if not tracking_numbers:
        url_addr = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipping_addresses"
        addrs = bc_get_json(auth_token, url_addr)
        if isinstance(addrs, list):
            for a in addrs:
                tn = (a.get("tracking_number", "") or "").strip()
                if tn != "":
                    tracking_numbers.append(tn)

    # Dedup (after normalization of whitespace)
    deduped: List[str] = []
    seen = set()
    for tn in tracking_numbers:
        tn2 = str(tn).strip()
        if tn2 not in seen:
            seen.add(tn2)
            deduped.append(tn2)

    return deduped, provider


def update_order_status(auth_token: str, store_id: str, order_id: int, new_status_id: int) -> None:
    url = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}"
    bc_put_json(auth_token, url, {"status_id": new_status_id})


# ----------------------------
# FedEx Track API
# ----------------------------
def fedex_get_access_token(base_url: str, client_id: str, client_secret: str) -> str:
    token_url = f"{base_url}/oauth/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}

    def _do():
        return requests.post(token_url, headers=headers, data=data, timeout=30)

    resp = retry_request(_do)
    resp.raise_for_status()
    j = resp.json()
    token = j.get("access_token")
    if not token:
        raise RuntimeError(f"FedEx token response missing access_token. Response: {json.dumps(j)[:500]}")
    return token


def fedex_track_bulk(base_url: str, token: str, tracking_numbers: List[str]) -> Dict[str, Any]:
    url = f"{base_url}/track/v1/trackingnumbers"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {
        "includeDetailedScans": True,
        "trackingInfo": [{"trackingNumberInfo": {"trackingNumber": tn}} for tn in tracking_numbers],
    }

    def _do():
        return requests.post(url, headers=headers, json=payload, timeout=30)

    resp = retry_request(_do)
    resp.raise_for_status()
    return resp.json()


@dataclass
class FedExTrackSummary:
    tracking_number: str
    latest_status: str
    delivered_at_utc: Optional[str]


def fedex_summarize_one(tr: Dict[str, Any], fallback_tracking_number: str) -> FedExTrackSummary:
    tn = fallback_tracking_number
    tni = (tr.get("trackingNumberInfo") or {})
    if isinstance(tni, dict) and tni.get("trackingNumber"):
        tn = str(tni["trackingNumber"])

    latest = tr.get("latestStatusDetail") or {}
    latest_status = latest.get("description") or latest.get("statusByLocale") or latest.get("code") or "Unknown"

    delivered_at: Optional[datetime] = None
    for item in (tr.get("dateAndTimes") or []):
        if isinstance(item, dict) and item.get("type") == "ACTUAL_DELIVERY":
            dt = _parse_dt(item.get("dateTime"))
            if dt:
                delivered_at = dt

    delivered_at_iso = delivered_at.astimezone(timezone.utc).isoformat() if delivered_at else None
    return FedExTrackSummary(tracking_number=tn, latest_status=str(latest_status), delivered_at_utc=delivered_at_iso)


def fedex_is_delivered(summary: FedExTrackSummary) -> bool:
    if summary.delivered_at_utc:
        return True
    return "delivered" in summary.latest_status.lower()


# ----------------------------
# Shipping-method rules
# ----------------------------
def is_ship_by_weight(shipping_method: str) -> bool:
    sm = normalize(shipping_method)
    return sm == "ship by weight" or "ship by weight" in sm


# ----------------------------
# State
# ----------------------------
def load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"version": 4, "orders": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: Dict[str, Any]) -> None:
    _safe_mkdir_for_file(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _md_escape_pipes(s: str) -> str:
    return (s or "").replace("|", "\\|")


def write_alert_markdown(alert_path: str, title: str, intro: str, rows: List[Dict[str, Any]]) -> None:
    _safe_mkdir_for_file(alert_path)
    lines: List[str] = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(intro)
    lines.append("")
    if rows:
        headers = ["Order ID", "Shipping Method", "Tracking #", "Reason", "Notes"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|---:|---|---|---|---|")
        for r in rows:
            oid = r.get("order_id", "")
            sm = _md_escape_pipes(str(r.get("shipping_method", "")))
            tn = _md_escape_pipes(str(r.get("tracking_number", "")))
            reason = _md_escape_pipes(str(r.get("reason", "")))
            notes = _md_escape_pipes(str(r.get("notes", "")))
            lines.append(f"| {oid} | {sm} | {tn} | {reason} | {notes} |")
        lines.append("")
    with open(alert_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    log_level = _get_env("LOG_LEVEL", default="INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # Required BC envs
    store_id = _get_env("BC_STORE_ID", required=True).strip()
    bc_auth_token = _get_env("BC_AUTH_TOKEN", required=True).strip()

    shipped_status_id = 2
    completed_status_id = 10

    fedex_client_id = _get_env("FEDEX_CLIENT_ID", required=True)
    fedex_client_secret = _get_env("FEDEX_CLIENT_SECRET", required=True)
    fedex_env = _get_env("FEDEX_ENV", default="prod").strip().lower()
    if fedex_env not in FEDEX_BASE_URLS:
        raise RuntimeError("FEDEX_ENV must be 'prod' or 'sandbox'")

    state_path = _get_env("STATE_PATH", default=".cache/shipped_state.json")

    # Alerts
    missing_tracking_path = _get_env("ALERT_PATH", default=".cache/missing_tracking.md")
    invalid_tracking_path = _get_env("INVALID_TRACKING_ALERT_PATH", default=".cache/invalid_tracking.md")
    fedex_empty_path = _get_env("FEDEX_EMPTY_ALERT_PATH", default=".cache/fedex_empty.md")

    min_fedex_interval = int(_get_env("MIN_FEDEX_CHECK_INTERVAL_SEC", default="600"))
    max_orders = int(_get_env("MAX_ORDERS_PER_RUN", default="0"))
    dry_run = _parse_bool(_get_env("DRY_RUN", default="false"))

    base_url = FEDEX_BASE_URLS[fedex_env]
    now = _utc_now()

    state = load_state(state_path)
    if "orders" not in state or not isinstance(state["orders"], dict):
        state["orders"] = {}
    cached_orders: Dict[str, Any] = state["orders"]

    logging.info("Fetching BigCommerce orders in status_id=2 (Shipped)...")
    orders = list_orders_by_status(bc_auth_token, store_id, shipped_status_id, limit=250)
    shipped_ids = [int(o.get("id")) for o in orders if isinstance(o, dict) and o.get("id") is not None]
    shipped_ids_set = set(shipped_ids)
    logging.info(f"Found {len(shipped_ids)} shipped orders.")

    # prune cache (removes itself once order leaves Shipped)
    for oid_str in list(cached_orders.keys()):
        try:
            oid = int(oid_str)
        except Exception:
            cached_orders.pop(oid_str, None)
            continue
        if oid not in shipped_ids_set:
            cached_orders.pop(oid_str, None)

    # hydrate new shipped orders
    hydrated = 0
    for oid in shipped_ids:
        if max_orders and hydrated >= max_orders:
            break
        oid_key = str(oid)
        if oid_key in cached_orders:
            continue

        shipping_method = fetch_shipping_method(bc_auth_token, store_id, oid)
        tracking_numbers, provider = fetch_tracking_numbers_and_provider(bc_auth_token, store_id, oid)

        cached_orders[oid_key] = {
            "order_id": oid,
            "shipping_method": shipping_method,
            "ship_by_weight": is_ship_by_weight(shipping_method),
            "tracking_numbers": tracking_numbers,   # may include weird values like "1"
            "shipping_provider": provider,
            "last_fedex_check_utc": None,
            "tracking_status": {},
            "tracking_delivered": {},
        }
        hydrated += 1

    if hydrated:
        logging.info(f"Hydrated {hydrated} new shipped orders into cache.")

    # ----------------------------
    # Alert detection (missing + invalid)
    # ----------------------------
    missing_tracking_rows: List[Dict[str, Any]] = []
    invalid_tracking_rows: List[Dict[str, Any]] = []

    for _, info in cached_orders.items():
        if info.get("ship_by_weight") is not True:
            continue

        tns = info.get("tracking_numbers") or []
        oid = int(info.get("order_id"))
        sm = info.get("shipping_method") or ""

        usable = [str(t).strip() for t in tns if str(t).strip() != ""]
        if not usable:
            missing_tracking_rows.append({
                "order_id": oid,
                "shipping_method": sm,
                "tracking_number": "",
                "reason": "missing",
                "notes": "Ship By Weight but no tracking numbers found in /shipments or /shipping_addresses.",
            })
            continue

        for tn in usable:
            is_bad, reason = classify_tracking_number(tn)
            if is_bad:
                invalid_tracking_rows.append({
                    "order_id": oid,
                    "shipping_method": sm,
                    "tracking_number": tn,
                    "reason": reason,
                    "notes": "Obvious invalid tracking format (will block completion).",
                })

    # Write missing tracking alert
    if missing_tracking_rows:
        logging.warning(f"ALERT: {len(missing_tracking_rows)} Ship By Weight shipped orders have no tracking number.")
        write_alert_markdown(
            missing_tracking_path,
            title="Ship By Weight orders missing tracking number",
            intro=f"Found **{len(missing_tracking_rows)}** shipped order(s) with shipping method **Ship By Weight** but no tracking number.",
            rows=missing_tracking_rows,
        )
        _write_step_output("missing_tracking", "true")
        _write_step_output("missing_tracking_count", str(len(missing_tracking_rows)))
    else:
        _write_step_output("missing_tracking", "false")
        _write_step_output("missing_tracking_count", "0")

    # Write invalid tracking alert
    if invalid_tracking_rows:
        logging.warning(f"ALERT: {len(invalid_tracking_rows)} invalid tracking number(s) detected on Ship By Weight orders.")
        write_alert_markdown(
            invalid_tracking_path,
            title="Ship By Weight orders with invalid tracking numbers",
            intro=f"Found **{len(invalid_tracking_rows)}** invalid tracking number(s) on shipped orders with shipping method **Ship By Weight**.",
            rows=invalid_tracking_rows,
        )
        _write_step_output("invalid_tracking", "true")
        _write_step_output("invalid_tracking_count", str(len(invalid_tracking_rows)))
    else:
        _write_step_output("invalid_tracking", "false")
        _write_step_output("invalid_tracking_count", "0")

    # Default for fedex_empty outputs
    _write_step_output("fedex_empty", "false")
    _write_step_output("fedex_empty_count", "0")

    # ----------------------------
    # 1) Complete anything NOT Ship By Weight immediately
    # ----------------------------
    to_complete_immediately: List[int] = []
    for _, info in list(cached_orders.items()):
        if info.get("ship_by_weight") is False:
            to_complete_immediately.append(int(info["order_id"]))

    if to_complete_immediately:
        logging.info(f"Immediate completion (NOT Ship By Weight): {len(to_complete_immediately)} orders.")
        for oid in to_complete_immediately:
            if dry_run:
                logging.info(f"[DRY_RUN] Would set order {oid} -> status_id={completed_status_id}")
            else:
                try:
                    update_order_status(bc_auth_token, store_id, oid, completed_status_id)
                    logging.info(f"Set order {oid} -> Completed.")
                except Exception as e:
                    logging.error(f"Failed to set order {oid} -> Completed: {e}")
                    continue
            cached_orders.pop(str(oid), None)

    # ----------------------------
    # 2) Ship By Weight => FedEx check (but NEVER complete if missing/invalid exists)
    # ----------------------------
    blocked_order_ids = set()
    for r in missing_tracking_rows:
        blocked_order_ids.add(int(r["order_id"]))
    for r in invalid_tracking_rows:
        blocked_order_ids.add(int(r["order_id"]))

    tns_to_check: List[str] = []
    tn_to_orders: Dict[str, List[int]] = {}

    for _, info in cached_orders.items():
        if info.get("ship_by_weight") is not True:
            continue

        oid = int(info["order_id"])
        if oid in blocked_order_ids:
            continue

        tns = [str(t).strip() for t in (info.get("tracking_numbers") or []) if str(t).strip() != ""]
        if not tns:
            continue

        provider = str(info.get("shipping_provider") or "").strip().lower()
        if provider and "fedex" not in provider:
            continue

        last_check = info.get("last_fedex_check_utc")
        last_check_dt = _parse_dt(last_check) if isinstance(last_check, str) else None
        if last_check_dt and (now - last_check_dt).total_seconds() < min_fedex_interval:
            continue

        for tn in tns:
            is_bad, _ = classify_tracking_number(tn)
            if is_bad:
                blocked_order_ids.add(oid)
                continue

            tns_to_check.append(tn)
            tn_to_orders.setdefault(tn, []).append(oid)

    uniq_tns: List[str] = []
    seen = set()
    for tn in tns_to_check:
        if tn not in seen:
            seen.add(tn)
            uniq_tns.append(tn)

    fedex_empty_batches = 0

    if uniq_tns:
        logging.info(f"FedEx checks needed this run (tracking #s): {len(uniq_tns)}")
        token = fedex_get_access_token(base_url, fedex_client_id, fedex_client_secret)

        tn_delivered: Dict[str, bool] = {}
        tn_status: Dict[str, str] = {}
        tn_seen_in_response: set[str] = set()

        for batch in chunks(uniq_tns, 30):
            try:
                resp = fedex_track_bulk(base_url, token, batch)
            except Exception as e:
                logging.error(f"FedEx track call failed for batch {batch}: {e}")
                continue

            output = resp.get("output", {})
            complete = output.get("completeTrackResults") or []

            if not isinstance(complete, list) or not complete:
                fedex_empty_batches += 1
                logging.warning("FedEx response had no completeTrackResults; skipping this batch.")
                continue

            for ctr in complete:
                tr_list = (ctr.get("trackResults") or [])
                if not tr_list:
                    continue
                tr0 = tr_list[0]

                tn_guess = ""
                tni = (tr0.get("trackingNumberInfo") or {})
                if isinstance(tni, dict):
                    tn_guess = str(tni.get("trackingNumber") or "").strip()
                if not tn_guess:
                    continue

                summary = fedex_summarize_one(tr0, tn_guess)
                delivered = fedex_is_delivered(summary)

                tn_seen_in_response.add(summary.tracking_number)
                tn_delivered[summary.tracking_number] = delivered
                tn_status[summary.tracking_number] = summary.latest_status

        fedex_unrecognized_rows: List[Dict[str, Any]] = []
        for tn in uniq_tns:
            tn_norm = tn.strip()
            tn_alt = tn_norm.replace(" ", "").replace("-", "")
            if tn_norm in tn_seen_in_response or tn_alt in tn_seen_in_response:
                continue

            for oid in tn_to_orders.get(tn, []):
                fedex_unrecognized_rows.append({
                    "order_id": oid,
                    "shipping_method": cached_orders.get(str(oid), {}).get("shipping_method", ""),
                    "tracking_number": tn,
                    "reason": "unrecognized_by_fedex",
                    "notes": "FedEx API did not return any trackResults for this tracking number.",
                })
                blocked_order_ids.add(oid)

        if fedex_unrecognized_rows:
            combined = invalid_tracking_rows + fedex_unrecognized_rows
            dedup = []
            seen_keys = set()
            for r in combined:
                k = (int(r.get("order_id", 0)), str(r.get("tracking_number", "")), str(r.get("reason", "")))
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                dedup.append(r)

            logging.warning(f"ALERT: {len(fedex_unrecognized_rows)} tracking number(s) were unrecognized by FedEx.")
            write_alert_markdown(
                invalid_tracking_path,
                title="Ship By Weight orders with invalid tracking numbers",
                intro=f"Found **{len(dedup)}** invalid/unrecognized tracking number(s) on shipped orders with shipping method **Ship By Weight**.",
                rows=dedup,
            )
            _write_step_output("invalid_tracking", "true")
            _write_step_output("invalid_tracking_count", str(len(dedup)))

        if fedex_empty_batches > 0:
            write_alert_markdown(
                fedex_empty_path,
                title="FedEx API returned empty tracking results for one or more batches",
                intro=f"FedEx returned empty `completeTrackResults` for **{fedex_empty_batches}** batch(es). This can be transient or indicate bad tracking input.",
                rows=[],
            )
            _write_step_output("fedex_empty", "true")
            _write_step_output("fedex_empty_count", str(fedex_empty_batches))

        to_complete_after_fedex: List[int] = []

        for _, info in cached_orders.items():
            if info.get("ship_by_weight") is not True:
                continue

            oid = int(info["order_id"])
            if oid in blocked_order_ids:
                continue

            tns = [str(t).strip() for t in (info.get("tracking_numbers") or []) if str(t).strip() != ""]
            if not tns:
                continue

            info["last_fedex_check_utc"] = now.isoformat()
            tracking_delivered = info.get("tracking_delivered") or {}
            tracking_status = info.get("tracking_status") or {}

            for tn in tns:
                tn_clean = tn.strip()
                if not tn_clean:
                    continue

                delivered = tn_delivered.get(tn_clean)
                status = tn_status.get(tn_clean)

                if delivered is None:
                    tn_nospace = tn_clean.replace(" ", "").replace("-", "")
                    delivered = tn_delivered.get(tn_nospace)
                    status = status or tn_status.get(tn_nospace)

                if delivered is not None:
                    tracking_delivered[tn_clean] = bool(delivered)
                if status is not None:
                    tracking_status[tn_clean] = str(status)

            info["tracking_delivered"] = tracking_delivered
            info["tracking_status"] = tracking_status

            all_delivered = True
            for tn in tns:
                if tracking_delivered.get(tn) is not True:
                    all_delivered = False
                    break

            if all_delivered:
                to_complete_after_fedex.append(oid)
                logging.info(f"All tracking numbers delivered => complete order {oid} (tns={tns})")

        if to_complete_after_fedex:
            logging.info(f"Marking {len(to_complete_after_fedex)} Ship By Weight orders Completed (ALL tns delivered).")
            for oid in to_complete_after_fedex:
                if dry_run:
                    logging.info(f"[DRY_RUN] Would set order {oid} -> status_id={completed_status_id}")
                else:
                    try:
                        update_order_status(bc_auth_token, store_id, oid, completed_status_id)
                        logging.info(f"Set order {oid} -> Completed.")
                    except Exception as e:
                        logging.error(f"Failed to set order {oid} -> Completed: {e}")
                        continue
                cached_orders.pop(str(oid), None)

    # Save state
    state["version"] = 4
    state["last_run_utc"] = now.isoformat()
    state["orders"] = cached_orders
    save_state(state_path, state)
    logging.info(f"Saved state to {state_path}. Cached shipped orders remaining: {len(cached_orders)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

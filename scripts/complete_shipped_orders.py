#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
complete_shipped_orders.py

Rules (status_id=2 only):
1) If shipping method is NOT "Ship By Weight" => move to status_id=10 (Completed) immediately.
2) If shipping method IS "Ship By Weight":
   - Pull all tracking numbers from the order (multiple shipments supported).
   - If there are NO tracking numbers: ALERT (but do NOT fail the run).
   - Use FedEx Track API to check ALL tracking numbers.
   - Only move the BigCommerce order to Completed when ALL tracking numbers are delivered.

Alerting (non-fatal):
A) Ship By Weight orders missing tracking numbers:
   - Write markdown file (MISSING_TRACKING_ALERT_PATH)
   - Set step output: missing_tracking=true
B) FedEx returned an "empty" response for the batch (no completeTrackResults / unusable results):
   - Write markdown file (FEDEX_EMPTY_ALERT_PATH)
   - Set step output: fedex_empty=true
C) Invalid tracking numbers detected (FedEx says not found/invalid/error for a tracking number):
   - Write markdown file (INVALID_TRACKING_ALERT_PATH)
   - Set step output: invalid_tracking=true

Caching:
- Each run lists shipped orders.
- For NOT Ship By Weight orders we do NOT call /shipments at all.
- Ship By Weight orders are cached with tracking numbers and rechecked on an interval.
"""

from __future__ import annotations

import json
import logging
import os
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
# Utilities
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
        yield lst[i : i + n]


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
        pass


def normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def is_ship_by_weight(shipping_method: str) -> bool:
    sm = normalize(shipping_method)
    return sm == "ship by weight" or "ship by weight" in sm


def _parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


# ----------------------------
# BigCommerce
# ----------------------------
def bc_headers() -> Dict[str, str]:
    return {
        "X-Auth-Token": _get_env("BC_AUTH_TOKEN", required=True),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def bc_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    """
    BigCommerce GET helper with clearer errors when BigCommerce returns non-JSON (HTML/empty/etc).
    """
    def _do():
        return requests.get(url, headers=bc_headers(), params=params, timeout=timeout)

    resp = retry_request(_do)

    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"BigCommerce GET failed {resp.status_code} {url}: {resp.text[:500]}",
            response=resp,
        )

    if not resp.text or not resp.text.strip():
        logging.warning(f"BigCommerce GET returned empty body for {url} (treating as empty list).")
        return []

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "json" not in ctype:
        raise RuntimeError(
            f"BigCommerce returned non-JSON for {url} (Content-Type={ctype}). "
            f"First 500 chars: {resp.text[:500]}"
        )

    return resp.json()


def bc_put_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Any:
    def _do():
        return requests.put(url, headers=bc_headers(), json=payload, timeout=timeout)

    resp = retry_request(_do)

    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"BigCommerce PUT failed {resp.status_code} {url}: {resp.text[:500]}",
            response=resp,
        )

    if not resp.text or not resp.text.strip():
        return {"status_code": resp.status_code}

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "json" in ctype:
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text[:500]}

    return {"status_code": resp.status_code, "text": resp.text[:500]}


def list_orders_by_status(store_id: str, status_id: int, limit: int = 250) -> List[Dict[str, Any]]:
    orders: List[Dict[str, Any]] = []
    page = 1

    while True:
        url = f"{BC_BASE_URL}/{store_id}/v2/orders"
        batch = bc_get_json(url, params={"status_id": status_id, "page": page, "limit": limit})
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected BigCommerce orders response (expected list). Got: {type(batch)}")
        if not batch:
            break
        orders.extend(batch)
        if len(batch) < limit:
            break
        page += 1

    return orders


def fetch_shipping_method(store_id: str, order_id: int) -> str:
    url = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipping_addresses"
    addrs = bc_get_json(url)
    if isinstance(addrs, list) and addrs:
        return (addrs[0].get("shipping_method", "") or "").strip()
    return ""


def fetch_tracking_numbers_and_provider(store_id: str, order_id: int) -> Tuple[List[str], str]:
    tracking_numbers: List[str] = []
    provider = ""

    url_ship = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipments"
    shipments = bc_get_json(url_ship)

    if isinstance(shipments, list):
        for sh in shipments:
            tn = (sh.get("tracking_number", "") or "").strip()
            if tn:
                tracking_numbers.append(tn)
            if not provider:
                provider = (sh.get("shipping_provider", "") or "").strip()

    if not tracking_numbers:
        url_addr = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipping_addresses"
        addrs = bc_get_json(url_addr)
        if isinstance(addrs, list):
            for a in addrs:
                tn = (a.get("tracking_number", "") or "").strip()
                if tn:
                    tracking_numbers.append(tn)

    deduped: List[str] = []
    seen = set()
    for tn in tracking_numbers:
        tn2 = tn.strip()
        if tn2 and tn2 not in seen:
            seen.add(tn2)
            deduped.append(tn2)

    return deduped, provider


def update_order_status(store_id: str, order_id: int, new_status_id: int) -> None:
    url = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}"
    bc_put_json(url, {"status_id": new_status_id})


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


def _looks_invalid_fedex_error(err: Any) -> bool:
    """
    FedEx errors vary; treat common not-found/invalid markers as invalid tracking.
    """
    if not isinstance(err, dict):
        return False
    code = str(err.get("code") or "").lower()
    msg = str(err.get("message") or "").lower()
    combined = f"{code} {msg}"
    markers = [
        "notfound", "not found", "no record", "invalid", "unrecognized", "does not exist",
        "trackingnumber.notfound", "tracking number not found",
    ]
    return any(m in combined for m in markers)


def _extract_tracking_number_from_any(obj: Any) -> str:
    """
    Best-effort extraction of tracking number from various FedEx response shapes.
    """
    if not isinstance(obj, dict):
        return ""
    # Common places:
    tni = obj.get("trackingNumberInfo")
    if isinstance(tni, dict):
        tn = str(tni.get("trackingNumber") or "").strip()
        if tn:
            return tn
    tn = str(obj.get("trackingNumber") or "").strip()
    if tn:
        return tn
    # Sometimes nested:
    if isinstance(obj.get("trackingNumber"), dict):
        tn2 = str(obj["trackingNumber"].get("trackingNumber") or "").strip()
        if tn2:
            return tn2
    return ""


# ----------------------------
# State / Alerts
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


def write_markdown_table_alert(path: str, title: str, intro: str, rows: List[Dict[str, str]]) -> None:
    _safe_mkdir_for_file(path)
    lines: List[str] = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(intro)
    lines.append("")
    lines.append("| Order ID | Tracking # | Details |")
    lines.append("|---:|---|---|")
    for r in rows:
        oid = str(r.get("order_id", "")).replace("|", "\\|")
        tn = str(r.get("tracking_number", "")).replace("|", "\\|")
        details = str(r.get("details", "")).replace("|", "\\|")
        lines.append(f"| {oid} | {tn} | {details} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_missing_tracking_markdown(alert_path: str, missing: List[Dict[str, Any]]) -> None:
    _safe_mkdir_for_file(alert_path)
    lines: List[str] = []
    lines.append("### Ship By Weight orders missing tracking number")
    lines.append("")
    lines.append(f"Found **{len(missing)}** shipped order(s) with shipping method **Ship By Weight** but no tracking number.")
    lines.append("")
    lines.append("| Order ID | Shipping Method | Notes |")
    lines.append("|---:|---|---|")
    for m in missing:
        oid = m.get("order_id")
        sm = (m.get("shipping_method") or "").replace("|", "\\|")
        notes = (m.get("notes") or "").replace("|", "\\|")
        lines.append(f"| {oid} | {sm} | {notes} |")
    lines.append("")
    lines.append("Action: Add tracking number to the shipment in BigCommerce (or adjust automation rules if this is expected).")
    with open(alert_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    log_level = _get_env("LOG_LEVEL", default="INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    store_id = _get_env("BC_STORE_ID", required=True)
    shipped_status_id = 2
    completed_status_id = 10

    fedex_client_id = _get_env("FEDEX_CLIENT_ID", required=True)
    fedex_client_secret = _get_env("FEDEX_CLIENT_SECRET", required=True)
    fedex_env = _get_env("FEDEX_ENV", default="prod").strip().lower()
    if fedex_env not in FEDEX_BASE_URLS:
        raise RuntimeError("FEDEX_ENV must be 'prod' or 'sandbox'")

    state_path = _get_env("STATE_PATH", default=".cache/shipped_state.json")

    # Existing missing-tracking alert file:
    missing_tracking_alert_path = _get_env("ALERT_PATH", default=".cache/missing_tracking.md")

    # New alert files:
    fedex_empty_alert_path = _get_env("FEDEX_EMPTY_ALERT_PATH", default=".cache/fedex_empty.md")
    invalid_tracking_alert_path = _get_env("INVALID_TRACKING_ALERT_PATH", default=".cache/invalid_tracking.md")

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
    orders = list_orders_by_status(store_id, shipped_status_id, limit=250)
    shipped_ids = [int(o.get("id")) for o in orders if isinstance(o, dict) and o.get("id") is not None]
    shipped_ids_set = set(shipped_ids)
    logging.info(f"Found {len(shipped_ids)} shipped orders.")

    # prune cache for orders that are no longer in shipped
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

        shipping_method = fetch_shipping_method(store_id, oid)
        ship_by_weight = is_ship_by_weight(shipping_method)

        if not ship_by_weight:
            cached_orders[oid_key] = {
                "order_id": oid,
                "shipping_method": shipping_method,
                "ship_by_weight": False,
                "tracking_numbers": [],
                "shipping_provider": "",
                "last_fedex_check_utc": None,
                "tracking_status": {},
                "tracking_delivered": {},
                "tracking_invalid": {},
            }
            hydrated += 1
            continue

        try:
            tracking_numbers, provider = fetch_tracking_numbers_and_provider(store_id, oid)
        except Exception as e:
            logging.error(f"Failed to fetch shipments/tracking for order {oid}: {e}")
            tracking_numbers, provider = [], ""

        cached_orders[oid_key] = {
            "order_id": oid,
            "shipping_method": shipping_method,
            "ship_by_weight": True,
            "tracking_numbers": tracking_numbers,
            "shipping_provider": provider,
            "last_fedex_check_utc": None,
            "tracking_status": {},
            "tracking_delivered": {},
            "tracking_invalid": {},  # tn -> {is_invalid: bool, reason: str}
        }
        hydrated += 1

    if hydrated:
        logging.info(f"Hydrated {hydrated} new shipped orders into cache.")

    # ALERT 1: Ship By Weight + no tracking
    missing_tracking_orders: List[Dict[str, Any]] = []
    for _, info in cached_orders.items():
        if info.get("ship_by_weight") is True:
            tns = info.get("tracking_numbers") or []
            if not tns:
                missing_tracking_orders.append(
                    {
                        "order_id": int(info.get("order_id")),
                        "shipping_method": info.get("shipping_method"),
                        "notes": "Ship By Weight but no tracking numbers found in /shipments or /shipping_addresses.",
                    }
                )

    if missing_tracking_orders:
        logging.warning(f"ALERT: {len(missing_tracking_orders)} Ship By Weight shipped orders have no tracking number.")
        write_missing_tracking_markdown(missing_tracking_alert_path, missing_tracking_orders)
        _write_step_output("missing_tracking", "true")
        _write_step_output("missing_tracking_count", str(len(missing_tracking_orders)))
    else:
        _write_step_output("missing_tracking", "false")
        _write_step_output("missing_tracking_count", "0")

    # 1) Complete anything that is NOT Ship By Weight immediately
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
                    update_order_status(store_id, oid, completed_status_id)
                    logging.info(f"Set order {oid} -> Completed.")
                except Exception as e:
                    logging.error(f"Failed to set order {oid} -> Completed: {e}")
                    continue
            cached_orders.pop(str(oid), None)

    # 2) Ship By Weight => FedEx check ALL tns; complete only if ALL delivered
    # Build list of tracking numbers to check this run AND a reverse index for alerts.
    tns_to_check: List[str] = []
    tn_to_orders: Dict[str, List[int]] = {}

    for _, info in cached_orders.items():
        if info.get("ship_by_weight") is not True:
            continue

        tns = info.get("tracking_numbers") or []
        if not tns:
            continue

        provider = str(info.get("shipping_provider") or "").strip().lower()
        if provider and "fedex" not in provider:
            continue

        last_check = info.get("last_fedex_check_utc")
        last_check_dt = _parse_dt(last_check) if isinstance(last_check, str) else None
        if last_check_dt and (now - last_check_dt).total_seconds() < min_fedex_interval:
            continue

        oid = int(info.get("order_id"))
        for tn in tns:
            tn_clean = str(tn).strip()
            if tn_clean:
                tns_to_check.append(tn_clean)
                tn_to_orders.setdefault(tn_clean, []).append(oid)

    uniq_tns: List[str] = []
    seen = set()
    for tn in tns_to_check:
        if tn not in seen:
            seen.add(tn)
            uniq_tns.append(tn)

    # Alert collectors
    fedex_empty_rows: List[Dict[str, str]] = []
    invalid_rows: List[Dict[str, str]] = []

    if uniq_tns:
        logging.info(f"FedEx checks needed this run (tracking #s): {len(uniq_tns)}")

        # Get token (if token fails, we consider this effectively an "empty" state for this run)
        try:
            token = fedex_get_access_token(base_url, fedex_client_id, fedex_client_secret)
        except Exception as e:
            logging.error(f"FedEx token request failed: {e}")
            # Treat as empty for all tns we intended to check
            for tn in uniq_tns:
                for oid in tn_to_orders.get(tn, []):
                    fedex_empty_rows.append(
                        {"order_id": str(oid), "tracking_number": tn, "details": "FedEx token request failed; could not query tracking."}
                    )
            token = ""

        tn_delivered: Dict[str, bool] = {}
        tn_status: Dict[str, str] = {}
        tn_invalid: Dict[str, str] = {}  # tn -> reason
        tns_seen_in_response: set[str] = set()

        if token:
            for batch in chunks(uniq_tns, 30):
                try:
                    resp = fedex_track_bulk(base_url, token, batch)
                except Exception as e:
                    logging.error(f"FedEx track call failed for batch {batch}: {e}")
                    # This is an "empty" result for this batch
                    for tn in batch:
                        for oid in tn_to_orders.get(tn, []):
                            fedex_empty_rows.append({"order_id": str(oid), "tracking_number": tn, "details": f"FedEx request failed: {e}"})
                    continue

                output = resp.get("output", {})
                complete = output.get("completeTrackResults") or []

                # EMPTY response scenario (your log case)
                if not isinstance(complete, list) or not complete:
                    logging.warning("FedEx response had no completeTrackResults; marking as EMPTY for this batch.")
                    for tn in batch:
                        for oid in tn_to_orders.get(tn, []):
                            fedex_empty_rows.append({"order_id": str(oid), "tracking_number": tn, "details": "FedEx response had no completeTrackResults (empty/unusable)."})
                    continue

                # Parse results
                for ctr in complete:
                    tr_list = (ctr.get("trackResults") or [])
                    if not isinstance(tr_list, list) or not tr_list:
                        # Can't tie this to a tn reliably; treat as "empty-ish" for safety
                        continue

                    # Use the first track result (common case)
                    tr0 = tr_list[0]
                    tn = _extract_tracking_number_from_any(tr0) or _extract_tracking_number_from_any(ctr)
                    tn = (tn or "").strip()
                    if not tn:
                        continue

                    tns_seen_in_response.add(tn)

                    # Invalid tracking case
                    err = tr0.get("error")
                    if err and _looks_invalid_fedex_error(err):
                        reason = f"{err.get('code', 'ERROR')}: {err.get('message', '')}".strip()
                        tn_invalid[tn] = reason or "FedEx reports tracking number not found/invalid."
                        continue

                    # Normal status case
                    summary = fedex_summarize_one(tr0, tn)
                    delivered = fedex_is_delivered(summary)
                    tn_delivered[summary.tracking_number] = delivered
                    tn_status[summary.tracking_number] = summary.latest_status

                # Any tracking numbers in the batch that did not show up at all -> consider "empty/unusable" for those
                # (This also catches the common invalid case where FedEx returns no record but doesn't include an explicit error object.)
                for tn in batch:
                    if tn not in tns_seen_in_response and tn not in tn_invalid:
                        for oid in tn_to_orders.get(tn, []):
                            fedex_empty_rows.append({"order_id": str(oid), "tracking_number": tn, "details": "FedEx returned no usable result for this tracking number in this batch."})

        # Apply results back onto cached orders and decide completion
        to_complete_after_fedex: List[int] = []

        for _, info in cached_orders.items():
            if info.get("ship_by_weight") is not True:
                continue

            tns = info.get("tracking_numbers") or []
            if not tns:
                continue

            info["last_fedex_check_utc"] = now.isoformat()

            tracking_delivered = info.get("tracking_delivered") or {}
            tracking_status = info.get("tracking_status") or {}
            tracking_invalid_state = info.get("tracking_invalid") or {}

            # Update per tracking number
            for tn in tns:
                tn_clean = str(tn).strip()
                if not tn_clean:
                    continue

                # Delivered/status updates
                delivered = tn_delivered.get(tn_clean)
                status = tn_status.get(tn_clean)

                if delivered is None:
                    tn_nospace = tn_clean.replace(" ", "")
                    delivered = tn_delivered.get(tn_nospace)
                    status = status or tn_status.get(tn_nospace)

                if delivered is not None:
                    tracking_delivered[tn_clean] = bool(delivered)
                if status is not None:
                    tracking_status[tn_clean] = str(status)

                # Invalid tracking updates
                reason = tn_invalid.get(tn_clean)
                if reason is None:
                    tn_nospace = tn_clean.replace(" ", "")
                    reason = tn_invalid.get(tn_nospace)

                if reason is not None:
                    tracking_invalid_state[tn_clean] = {"is_invalid": True, "reason": str(reason)}
                    # Add to alert rows
                    oid = int(info.get("order_id"))
                    invalid_rows.append({"order_id": str(oid), "tracking_number": tn_clean, "details": str(reason)})

            info["tracking_delivered"] = tracking_delivered
            info["tracking_status"] = tracking_status
            info["tracking_invalid"] = tracking_invalid_state

            # Completion check: ALL must be delivered; invalid blocks completion naturally because delivered won't be True.
            all_delivered = True
            for tn in tns:
                tn_clean = str(tn).strip()
                if tn_clean and tracking_delivered.get(tn_clean) is not True:
                    all_delivered = False
                    break

            if all_delivered:
                to_complete_after_fedex.append(int(info["order_id"]))
                logging.info(f"All tracking numbers delivered => complete order {info['order_id']} (tns={tns})")

        if to_complete_after_fedex:
            logging.info(f"Marking {len(to_complete_after_fedex)} Ship By Weight orders Completed (ALL tns delivered).")
            for oid in to_complete_after_fedex:
                if dry_run:
                    logging.info(f"[DRY_RUN] Would set order {oid} -> status_id={completed_status_id}")
                else:
                    try:
                        update_order_status(store_id, oid, completed_status_id)
                        logging.info(f"Set order {oid} -> Completed.")
                    except Exception as e:
                        logging.error(f"Failed to set order {oid} -> Completed: {e}")
                        continue
                cached_orders.pop(str(oid), None)

    # --- Emit alerts for FedEx EMPTY responses ---
    if fedex_empty_rows:
        # Dedup rows
        dedup = {}
        for r in fedex_empty_rows:
            key = (r.get("order_id", ""), r.get("tracking_number", ""), r.get("details", ""))
            dedup[key] = r
        fedex_empty_rows = list(dedup.values())

        logging.warning(f"ALERT: FedEx returned empty/unusable results for {len(fedex_empty_rows)} order/tracking pairs.")
        write_markdown_table_alert(
            fedex_empty_alert_path,
            title="FedEx returned empty / unusable tracking results",
            intro=f"Found **{len(fedex_empty_rows)}** order/tracking pair(s) where FedEx returned empty or unusable results (no completeTrackResults or no usable trackResults).",
            rows=fedex_empty_rows,
        )
        _write_step_output("fedex_empty", "true")
        _write_step_output("fedex_empty_count", str(len(fedex_empty_rows)))
    else:
        _write_step_output("fedex_empty", "false")
        _write_step_output("fedex_empty_count", "0")

    # --- Emit alerts for INVALID tracking numbers ---
    if invalid_rows:
        dedup = {}
        for r in invalid_rows:
            key = (r.get("order_id", ""), r.get("tracking_number", ""), r.get("details", ""))
            dedup[key] = r
        invalid_rows = list(dedup.values())

        logging.warning(f"ALERT: Invalid FedEx tracking numbers detected: {len(invalid_rows)} order/tracking pairs.")
        write_markdown_table_alert(
            invalid_tracking_alert_path,
            title="Invalid FedEx tracking numbers detected",
            intro=f"Found **{len(invalid_rows)}** order/tracking pair(s) where FedEx reported the tracking number as invalid/not found (or returned an explicit error).",
            rows=invalid_rows,
        )
        _write_step_output("invalid_tracking", "true")
        _write_step_output("invalid_tracking_count", str(len(invalid_rows)))
    else:
        _write_step_output("invalid_tracking", "false")
        _write_step_output("invalid_tracking_count", "0")

    # Save state
    state["version"] = 4
    state["last_run_utc"] = now.isoformat()
    state["orders"] = cached_orders
    save_state(state_path, state)
    logging.info(f"Saved state to {state_path}. Cached shipped orders remaining: {len(cached_orders)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

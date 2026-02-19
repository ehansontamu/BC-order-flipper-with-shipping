#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
complete_shipped_orders.py

Rules (status_id=2 only):
1) If shipping method is NOT "Ship By Weight" => move to status_id=10 (Completed) immediately.
2) If shipping method IS "Ship By Weight":
   - Read ALL BigCommerce shipments (multiple shipments supported).
   - If ANY shipment is missing a tracking number => ALERT and DO NOT complete.
   - Use FedEx Track API to check ALL tracking numbers.
   - If ANY tracking number is invalid/untrackable => ALERT and DO NOT complete.
   - Only complete the BigCommerce order when ALL tracking numbers are delivered.

Alerting (non-fatal):
- missing tracking => missing_tracking.md
- invalid tracking => invalid_tracking.md
- FedEx empty response batches => fedex_empty.md

Caching:
- Each run lists shipped orders (status 2).
- Shipping method + shipment/tracking metadata are fetched once per order then cached.
- Cache is pruned when an order is no longer in status 2.

GitHub Actions step outputs:
- missing_tracking (true/false), missing_tracking_count
- invalid_tracking (true/false), invalid_tracking_count
- fedex_empty (true/false), fedex_empty_count
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


def chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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


def _safe_json(resp: requests.Response, context: str) -> Any:
    """
    BigCommerce sometimes returns empty/non-JSON bodies (esp. on errors or edge cases).
    This prevents JSONDecodeError crashes and gives you useful logs.
    """
    text = resp.text or ""
    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"{context} failed {resp.status_code}: {text[:500]}",
            response=resp,
        )

    # For safety, allow empty JSON-ish responses
    if text.strip() == "":
        # Treat as empty object rather than crashing
        logging.warning(f"{context}: empty response body (treated as empty JSON).")
        return {}

    try:
        return resp.json()
    except Exception:
        logging.error(f"{context}: response was not valid JSON. First 500 chars: {text[:500]}")
        raise


def bc_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    def _do():
        return requests.get(url, headers=bc_headers(), params=params, timeout=timeout)

    resp = retry_request(_do)
    return _safe_json(resp, f"BigCommerce GET {url}")


def bc_put_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Any:
    def _do():
        return requests.put(url, headers=bc_headers(), json=payload, timeout=timeout)

    resp = retry_request(_do)
    return _safe_json(resp, f"BigCommerce PUT {url}")


def list_orders_by_status(store_id: str, status_id: int, limit: int = 250) -> List[Dict[str, Any]]:
    orders: List[Dict[str, Any]] = []
    page = 1

    while True:
        url = f"{BC_BASE_URL}/{store_id}/v2/orders"
        batch = bc_get_json(url, params={"status_id": status_id, "page": page, "limit": limit})
        if not isinstance(batch, list):
            # Some BC errors return dicts; let’s show it
            raise RuntimeError(f"Unexpected BigCommerce orders response (expected list). Got: {type(batch)} -> {batch}")
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


@dataclass
class ShipmentInfo:
    shipment_id: Optional[int]
    tracking_number: str
    shipping_provider: str


def fetch_shipments_and_tracking(store_id: str, order_id: int) -> Tuple[List[str], str, int, int, List[ShipmentInfo]]:
    """
    Returns:
      tracking_numbers (deduped, non-empty)
      provider (first non-empty provider observed)
      shipment_count (how many shipment records exist)
      missing_tracking_count (shipment records with empty tracking_number)
      shipments_detail (for better alert logging/troubleshooting)
    """
    tracking_numbers: List[str] = []
    provider = ""
    shipment_count = 0
    missing_tracking_count = 0
    shipments_detail: List[ShipmentInfo] = []

    url_ship = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipments"
    shipments = bc_get_json(url_ship)

    if isinstance(shipments, list):
        shipment_count = len(shipments)
        for sh in shipments:
            tn_raw = (sh.get("tracking_number", "") or "").strip()
            prov = (sh.get("shipping_provider", "") or "").strip()
            sid = sh.get("id")
            if sid is not None:
                try:
                    sid = int(sid)
                except Exception:
                    sid = None

            shipments_detail.append(
                ShipmentInfo(
                    shipment_id=sid,
                    tracking_number=tn_raw,
                    shipping_provider=prov,
                )
            )

            if tn_raw:
                tracking_numbers.append(tn_raw)
            else:
                missing_tracking_count += 1

            if not provider and prov:
                provider = prov

    # Fallback only if there are NO shipment records at all
    if shipment_count == 0:
        url_addr = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipping_addresses"
        addrs = bc_get_json(url_addr)
        if isinstance(addrs, list):
            for a in addrs:
                tn = (a.get("tracking_number", "") or "").strip()
                if tn:
                    tracking_numbers.append(tn)

    # Dedupe
    deduped: List[str] = []
    seen = set()
    for tn in tracking_numbers:
        tn2 = tn.strip()
        if tn2 and tn2 not in seen:
            seen.add(tn2)
            deduped.append(tn2)

    return deduped, provider, shipment_count, missing_tracking_count, shipments_detail


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


def fedex_summarize_one(track_result: Dict[str, Any], fallback_tracking_number: str) -> FedExTrackSummary:
    tn = fallback_tracking_number
    tni = track_result.get("trackingNumberInfo") or {}
    if isinstance(tni, dict) and tni.get("trackingNumber"):
        tn = str(tni["trackingNumber"])

    latest = track_result.get("latestStatusDetail") or {}
    latest_status = latest.get("description") or latest.get("statusByLocale") or latest.get("code") or "Unknown"

    delivered_at: Optional[datetime] = None
    for item in (track_result.get("dateAndTimes") or []):
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


def normalize_tracking(tn: str) -> str:
    return re.sub(r"\s+", "", (tn or "").strip())


# ----------------------------
# Shipping-method rules
# ----------------------------
def normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


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


def write_markdown_table(alert_path: str, title: str, intro: str, rows: List[Dict[str, str]], columns: List[Tuple[str, str]]) -> None:
    """
    columns: list of (col_title, key_in_row_dict)
    """
    _safe_mkdir_for_file(alert_path)
    lines: List[str] = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(intro)
    lines.append("")
    # header
    lines.append("| " + " | ".join([c[0] for c in columns]) + " |")
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")
    # rows
    for r in rows:
        vals = []
        for _, key in columns:
            v = (r.get(key, "") or "").replace("|", "\\|")
            vals.append(v)
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    with open(alert_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    log_level = _get_env("LOG_LEVEL", default="INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    store_id = _get_env("BC_STORE_ID", required=True)
    shipped_status_id = 2
    completed_status_id = 10

    fedex_client_id = _get_env("FEDEX_CLIENT_ID", required=True)
    fedex_client_secret = _get_env("FEDEX_CLIENT_SECRET", required=True)
    fedex_env = _get_env("FEDEX_ENV", default="prod").strip().lower()
    if fedex_env not in FEDEX_BASE_URLS:
        raise RuntimeError("FEDEX_ENV must be 'prod' or 'sandbox'")

    state_path = _get_env("STATE_PATH", default=".cache/shipped_state.json")
    missing_tracking_alert_path = _get_env("ALERT_PATH", default=".cache/missing_tracking.md")
    fedex_empty_alert_path = _get_env("FEDEX_EMPTY_ALERT_PATH", default=".cache/fedex_empty.md")
    invalid_tracking_alert_path = _get_env("INVALID_TRACKING_ALERT_PATH", default=".cache/invalid_tracking.md")

    min_fedex_interval = int(_get_env("MIN_FEDEX_CHECK_INTERVAL_SEC", default="600"))
    max_orders = int(_get_env("MAX_ORDERS_PER_RUN", default="0"))
    dry_run = _parse_bool(_get_env("DRY_RUN", default="false"))

    base_url = FEDEX_BASE_URLS[fedex_env]
    now = _utc_now()

    # Default step outputs
    _write_step_output("missing_tracking", "false")
    _write_step_output("missing_tracking_count", "0")
    _write_step_output("invalid_tracking", "false")
    _write_step_output("invalid_tracking_count", "0")
    _write_step_output("fedex_empty", "false")
    _write_step_output("fedex_empty_count", "0")

    state = load_state(state_path)
    if "orders" not in state or not isinstance(state["orders"], dict):
        state["orders"] = {}
    cached_orders: Dict[str, Any] = state["orders"]

    logging.info("Fetching BigCommerce orders in status_id=2 (Shipped)...")
    orders = list_orders_by_status(store_id, shipped_status_id, limit=250)
    shipped_ids = [int(o.get("id")) for o in orders if isinstance(o, dict) and o.get("id") is not None]
    shipped_ids_set = set(shipped_ids)
    logging.info(f"Found {len(shipped_ids)} shipped orders.")

    # prune cache for orders no longer in shipped status
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
        tracking_numbers, provider, shipment_count, missing_tracking_count, shipments_detail = fetch_shipments_and_tracking(store_id, oid)

        cached_orders[oid_key] = {
            "order_id": oid,
            "shipping_method": shipping_method,
            "ship_by_weight": is_ship_by_weight(shipping_method),

            "tracking_numbers": tracking_numbers,  # deduped non-empty
            "shipping_provider": provider,

            # critical for multi-shipment correctness
            "shipment_count": shipment_count,
            "missing_tracking_count": missing_tracking_count,
            "shipments_detail": [
                {
                    "shipment_id": s.shipment_id,
                    "tracking_number": s.tracking_number,
                    "shipping_provider": s.shipping_provider,
                }
                for s in shipments_detail
            ],

            "last_fedex_check_utc": None,
            "tracking_status": {},
            "tracking_delivered": {},
            "invalid_tracking_numbers": [],
        }
        hydrated += 1
    if hydrated:
        logging.info(f"Hydrated {hydrated} new shipped orders into cache.")

    # ----------------------------------------
    # ALERT: Ship By Weight + missing tracking in ANY shipment
    # ----------------------------------------
    missing_tracking_orders: List[Dict[str, str]] = []
    for _, info in cached_orders.items():
        if info.get("ship_by_weight") is True:
            if int(info.get("missing_tracking_count") or 0) > 0:
                missing_tracking_orders.append(
                    {
                        "order_id": str(info.get("order_id")),
                        "shipping_method": str(info.get("shipping_method") or ""),
                        "notes": f"{info.get('missing_tracking_count')} shipment(s) missing tracking out of {info.get('shipment_count')} shipment(s).",
                    }
                )

    if missing_tracking_orders:
        logging.warning(f"ALERT: {len(missing_tracking_orders)} Ship By Weight shipped orders have missing tracking numbers on at least one shipment.")
        write_markdown_table(
            missing_tracking_alert_path,
            title="Ship By Weight orders missing tracking number",
            intro=f"Found **{len(missing_tracking_orders)}** shipped order(s) where shipping method is **Ship By Weight** but one or more shipments have no tracking number.",
            rows=missing_tracking_orders,
            columns=[("Order ID", "order_id"), ("Shipping Method", "shipping_method"), ("Notes", "notes")],
        )
        _write_step_output("missing_tracking", "true")
        _write_step_output("missing_tracking_count", str(len(missing_tracking_orders)))

    # ----------------------------------------
    # 1) Complete anything that is NOT Ship By Weight immediately
    # ----------------------------------------
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

    # ----------------------------------------
    # 2) Ship By Weight => FedEx check ALL tns; complete only if ALL delivered
    #    AND no missing shipment tracking AND no invalid tracking
    # ----------------------------------------
    # Build list of tracking numbers that need FedEx checking this run
    tns_to_check: List[str] = []

    for _, info in cached_orders.items():
        if info.get("ship_by_weight") is not True:
            continue

        # If ANY shipment missing tracking, block completion (and do not waste FedEx calls)
        if int(info.get("missing_tracking_count") or 0) > 0:
            continue

        tns = info.get("tracking_numbers") or []
        if not tns:
            # If shipments existed, missing would have been >0, but keep safe:
            continue

        provider = str(info.get("shipping_provider") or "").strip().lower()
        if provider and "fedex" not in provider:
            # Not FedEx (or unknown); do not attempt FedEx calls
            continue

        last_check = info.get("last_fedex_check_utc")
        last_check_dt = _parse_dt(last_check) if isinstance(last_check, str) else None
        if last_check_dt and (now - last_check_dt).total_seconds() < min_fedex_interval:
            continue

        for tn in tns:
            tn_clean = normalize_tracking(str(tn))
            if tn_clean:
                tns_to_check.append(tn_clean)

    # Dedup
    uniq_tns: List[str] = []
    seen = set()
    for tn in tns_to_check:
        if tn not in seen:
            seen.add(tn)
            uniq_tns.append(tn)

    fedex_empty_batches: List[Dict[str, str]] = []
    # Maps for results
    tn_delivered: Dict[str, bool] = {}
    tn_status: Dict[str, str] = {}
    tn_invalid: Dict[str, str] = {}  # tracking -> reason

    if uniq_tns:
        logging.info(f"FedEx checks needed this run (tracking #s): {len(uniq_tns)}")
        token = fedex_get_access_token(base_url, fedex_client_id, fedex_client_secret)

        for batch in chunks(uniq_tns, 30):
            try:
                resp = fedex_track_bulk(base_url, token, batch)
            except Exception as e:
                logging.error(f"FedEx track call failed for batch {batch}: {e}")
                # Mark these as unknown/invalid for this run, but don't permanently label them invalid.
                for tn in batch:
                    tn_invalid[tn] = f"FedEx request failed: {e}"
                continue

            output = resp.get("output", {})
            complete = output.get("completeTrackResults") or []

            if not isinstance(complete, list) or not complete:
                logging.warning("FedEx response had no completeTrackResults; skipping this batch.")
                fedex_empty_batches.append(
                    {
                        "batch": ", ".join(batch),
                        "notes": "FedEx response had no completeTrackResults for this request.",
                    }
                )
                # Treat all in this batch as invalid/untrackable for now (alerts) because they produced no usable results.
                for tn in batch:
                    tn_invalid[tn] = "FedEx response had no completeTrackResults"
                continue

            # Parse results & detect invalids:
            # If a TN appears in our request but FedEx gives no trackResults for it, mark invalid.
            returned_tns_in_this_batch = set()

            for ctr in complete:
                tr_list = (ctr.get("trackResults") or [])
                if not tr_list:
                    # There is a completeTrackResult but no trackResults — likely invalid.
                    # We don’t know which TN this was (FedEx sometimes includes info elsewhere),
                    # so we’ll skip marking here and rely on coverage check below.
                    continue

                tr0 = tr_list[0]
                # pull TN
                tn_guess = ""
                tni = (tr0.get("trackingNumberInfo") or {})
                if isinstance(tni, dict):
                    tn_guess = str(tni.get("trackingNumber") or "").strip()
                tn_guess = normalize_tracking(tn_guess)

                # If FedEx didn't echo back the tracking number, we can't map it safely
                if not tn_guess:
                    continue

                returned_tns_in_this_batch.add(tn_guess)

                # Some FedEx responses include error notifications on trackResults
                # If there is a severe notification, treat as invalid.
                notifications = tr0.get("notifications") or []
                severe = None
                if isinstance(notifications, list):
                    for n in notifications:
                        if isinstance(n, dict):
                            sev = str(n.get("severity") or "").upper()
                            msg = str(n.get("message") or "")
                            code = str(n.get("code") or "")
                            if sev in {"ERROR", "FAILURE"}:
                                severe = f"{sev} {code} {msg}".strip()
                                break
                if severe:
                    tn_invalid[tn_guess] = severe
                    tn_status[tn_guess] = "Invalid/Untrackable"
                    tn_delivered[tn_guess] = False
                    continue

                summary = fedex_summarize_one(tr0, tn_guess)
                delivered = fedex_is_delivered(summary)

                tn_delivered[summary.tracking_number] = delivered
                tn_status[summary.tracking_number] = summary.latest_status

            # Coverage check: any tracking in batch not returned by FedEx is invalid/untrackable
            for requested_tn in batch:
                req_norm = normalize_tracking(requested_tn)
                if req_norm and req_norm not in returned_tns_in_this_batch and req_norm not in tn_invalid:
                    tn_invalid[req_norm] = "No FedEx result returned for this tracking number"

    # ----------------------------------------
    # Apply FedEx results to cached orders & decide completion
    # ----------------------------------------
    invalid_tracking_orders: List[Dict[str, str]] = []

    if uniq_tns:
        for _, info in cached_orders.items():
            if info.get("ship_by_weight") is not True:
                continue

            # Block: missing tracking in any shipment
            if int(info.get("missing_tracking_count") or 0) > 0:
                continue

            tns = info.get("tracking_numbers") or []
            if not tns:
                continue

            provider = str(info.get("shipping_provider") or "").strip().lower()
            if provider and "fedex" not in provider:
                continue

            info["last_fedex_check_utc"] = now.isoformat()
            tracking_delivered = info.get("tracking_delivered") or {}
            tracking_status = info.get("tracking_status") or {}
            invalid_list = set(info.get("invalid_tracking_numbers") or [])

            # Update per tracking
            for tn in tns:
                tn_norm = normalize_tracking(str(tn))
                if not tn_norm:
                    continue

                # Invalid?
                if tn_norm in tn_invalid:
                    invalid_list.add(tn_norm)
                    tracking_delivered[tn_norm] = False
                    tracking_status[tn_norm] = f"INVALID: {tn_invalid[tn_norm]}"
                    continue

                # Delivered/status?
                if tn_norm in tn_delivered:
                    tracking_delivered[tn_norm] = bool(tn_delivered[tn_norm])
                if tn_norm in tn_status:
                    tracking_status[tn_norm] = str(tn_status[tn_norm])

            info["tracking_delivered"] = tracking_delivered
            info["tracking_status"] = tracking_status
            info["invalid_tracking_numbers"] = sorted(invalid_list)

            # If any invalid tracking exists, alert and block completion
            if info["invalid_tracking_numbers"]:
                invalid_tracking_orders.append(
                    {
                        "order_id": str(info.get("order_id")),
                        "shipping_method": str(info.get("shipping_method") or ""),
                        "tracking_numbers": ", ".join([normalize_tracking(t) for t in tns]),
                        "notes": "Invalid/untrackable tracking detected: " + ", ".join(info["invalid_tracking_numbers"]),
                    }
                )

    # Write invalid tracking alert (if any)
    if invalid_tracking_orders:
        logging.warning(f"ALERT: {len(invalid_tracking_orders)} Ship By Weight shipped orders have invalid tracking numbers.")
        write_markdown_table(
            invalid_tracking_alert_path,
            title="Ship By Weight orders with invalid tracking numbers",
            intro=f"Found **{len(invalid_tracking_orders)}** shipped order(s) with **Ship By Weight** where one or more tracking numbers appear invalid/untrackable via FedEx.",
            rows=invalid_tracking_orders,
            columns=[
                ("Order ID", "order_id"),
                ("Shipping Method", "shipping_method"),
                ("Tracking Numbers", "tracking_numbers"),
                ("Notes", "notes"),
            ],
        )
        _write_step_output("invalid_tracking", "true")
        _write_step_output("invalid_tracking_count", str(len(invalid_tracking_orders)))

    # Write FedEx empty response alert (if any)
    if fedex_empty_batches:
        logging.warning(f"ALERT: {len(fedex_empty_batches)} FedEx request batch(es) returned empty completeTrackResults.")
        write_markdown_table(
            fedex_empty_alert_path,
            title="FedEx empty responses detected",
            intro=f"Found **{len(fedex_empty_batches)}** FedEx request batch(es) where the API returned no `completeTrackResults`.",
            rows=fedex_empty_batches,
            columns=[("Batch", "batch"), ("Notes", "notes")],
        )
        _write_step_output("fedex_empty", "true")
        _write_step_output("fedex_empty_count", str(len(fedex_empty_batches)))

    # Completion decision for Ship By Weight orders:
    to_complete_after_fedex: List[int] = []
    for _, info in cached_orders.items():
        if info.get("ship_by_weight") is not True:
            continue

        # Block completion if any shipment missing tracking
        if int(info.get("missing_tracking_count") or 0) > 0:
            continue

        # Block completion if any invalid tracking numbers
        if info.get("invalid_tracking_numbers"):
            continue

        tns = info.get("tracking_numbers") or []
        if not tns:
            continue

        provider = str(info.get("shipping_provider") or "").strip().lower()
        if provider and "fedex" not in provider:
            continue

        tracking_delivered = info.get("tracking_delivered") or {}
        # Must have ALL delivered
        all_delivered = True
        for tn in tns:
            tn_norm = normalize_tracking(str(tn))
            if tn_norm and tracking_delivered.get(tn_norm) is not True:
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

    # Persist state
    state["version"] = 4
    state["last_run_utc"] = now.isoformat()
    state["orders"] = cached_orders
    save_state(state_path, state)
    logging.info(f"Saved state to {state_path}. Cached shipped orders remaining: {len(cached_orders)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

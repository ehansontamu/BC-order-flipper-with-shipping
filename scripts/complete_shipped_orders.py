#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
complete_shipped_orders.py

Goal:
- Pull all BigCommerce orders in status_id=2 (Shipped).
- If shipping method is "On Campus Delivery" (no tracking), move to status_id=10 (Completed).
- If shipping is "Ship By Weight" and a FedEx tracking number exists, call FedEx Track API.
  If delivered, move to status_id=10 (Completed).

Caching / low BigCommerce calls:
- We still list the "Shipped" orders each run (pagination).
- For each shipped order, we only fetch shipping-method + tracking numbers from BigCommerce
  the *first* time we see that order (stored in STATE_PATH). After that, we reuse cached info.
- For FedEx, we optionally rate-limit checks per order via MIN_FEDEX_CHECK_INTERVAL_SEC.

Required env vars:
- BC_STORE_ID
- BC_AUTH_TOKEN
- FEDEX_CLIENT_ID
- FEDEX_CLIENT_SECRET

Optional env vars:
- FEDEX_ENV (prod|sandbox) default: prod
- STATE_PATH default: .cache/shipped_state.json
- MIN_FEDEX_CHECK_INTERVAL_SEC default: 600
- MAX_ORDERS_PER_RUN default: 0 (no cap)
- DRY_RUN default: false
- LOG_LEVEL default: INFO
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
    def _do():
        return requests.get(url, headers=bc_headers(), params=params, timeout=timeout)

    resp = retry_request(_do)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"BigCommerce GET failed {resp.status_code} {url}: {resp.text[:500]}", response=resp)
    return resp.json()


def bc_put_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Any:
    def _do():
        return requests.put(url, headers=bc_headers(), json=payload, timeout=timeout)

    resp = retry_request(_do)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"BigCommerce PUT failed {resp.status_code} {url}: {resp.text[:500]}", response=resp)

    try:
        return resp.json()
    except Exception:
        return {"status_code": resp.status_code, "text": resp.text}


def list_orders_by_status(store_id: str, status_id: int, limit: int = 250) -> List[Dict[str, Any]]:
    """
    Returns the full list of orders for the given status_id.
    Uses /v2/orders?status_id=...&page=...&limit=... (v2 pagination).
    """
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
    """
    Fetch shipping method from /v2/orders/{order_id}/shipping_addresses
    """
    url = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipping_addresses"
    addrs = bc_get_json(url)
    if isinstance(addrs, list) and addrs:
        return (addrs[0].get("shipping_method", "") or "").strip()
    return ""


def fetch_tracking_numbers_and_provider(store_id: str, order_id: int) -> Tuple[List[str], str]:
    """
    Fetch tracking numbers primarily from:
      GET /v2/orders/{order_id}/shipments

    Fallback (some setups put it on the shipping address):
      GET /v2/orders/{order_id}/shipping_addresses

    Returns: (tracking_numbers, shipping_provider_guess)
    """
    tracking_numbers: List[str] = []
    provider = ""

    # Primary: shipments
    url_ship = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipments"
    shipments = bc_get_json(url_ship)

    if isinstance(shipments, list):
        for sh in shipments:
            tn = (sh.get("tracking_number", "") or "").strip()
            if tn:
                tracking_numbers.append(tn)
            if not provider:
                provider = (sh.get("shipping_provider", "") or "").strip()

    # Fallback: shipping address may contain tracking_number
    if not tracking_numbers:
        url_addr = f"{BC_BASE_URL}/{store_id}/v2/orders/{order_id}/shipping_addresses"
        addrs = bc_get_json(url_addr)
        if isinstance(addrs, list):
            for a in addrs:
                tn = (a.get("tracking_number", "") or "").strip()
                if tn:
                    tracking_numbers.append(tn)

    # de-dupe preserving order
    deduped: List[str] = []
    seen = set()
    for tn in tracking_numbers:
        if tn not in seen:
            seen.add(tn)
            deduped.append(tn)

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
    """
    FedEx Track endpoint supports up to 30 tracking numbers per request.
    """
    url = f"{base_url}/track/v1/trackingnumbers"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {"includeDetailedScans": True, "trackingInfo": [{"trackingNumberInfo": {"trackingNumber": tn}} for tn in tracking_numbers]}

    def _do():
        return requests.post(url, headers=headers, json=payload, timeout=30)

    resp = retry_request(_do)
    resp.raise_for_status()
    return resp.json()


def _parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


@dataclass
class FedExTrackSummary:
    tracking_number: str
    latest_status: str
    delivered_at_utc: Optional[str]
    last_scan_at_utc: Optional[str]
    raw: Dict[str, Any]


def fedex_summarize_one(tr: Dict[str, Any], fallback_tracking_number: str) -> FedExTrackSummary:
    tn = fallback_tracking_number
    tni = (tr.get("trackingNumberInfo") or {})
    if isinstance(tni, dict) and tni.get("trackingNumber"):
        tn = str(tni["trackingNumber"])

    latest = tr.get("latestStatusDetail") or {}
    latest_status = (
        latest.get("description")
        or latest.get("statusByLocale")
        or latest.get("code")
        or "Unknown"
    )

    delivered_at: Optional[datetime] = None
    last_scan_at: Optional[datetime] = None

    for item in (tr.get("dateAndTimes") or []):
        if isinstance(item, dict) and item.get("type") == "ACTUAL_DELIVERY":
            dt = _parse_dt(item.get("dateTime"))
            if dt:
                delivered_at = dt

    scan_events = tr.get("scanEvents") or []
    if isinstance(scan_events, list) and scan_events:
        scan_events_sorted = sorted(
            scan_events,
            key=lambda e: _parse_dt(e.get("date")) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        last_scan_at = _parse_dt(scan_events_sorted[0].get("date"))

    delivered_at_iso = delivered_at.astimezone(timezone.utc).isoformat() if delivered_at else None
    last_scan_iso = last_scan_at.astimezone(timezone.utc).isoformat() if last_scan_at else None

    return FedExTrackSummary(
        tracking_number=tn,
        latest_status=str(latest_status),
        delivered_at_utc=delivered_at_iso,
        last_scan_at_utc=last_scan_iso,
        raw=tr,
    )


def fedex_is_delivered(summary: FedExTrackSummary) -> bool:
    if summary.delivered_at_utc:
        return True
    return "delivered" in summary.latest_status.lower()


# ----------------------------
# State
# ----------------------------
def load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"version": 1, "orders": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: Dict[str, Any]) -> None:
    _safe_mkdir_for_file(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def normalize_shipping_method(s: str) -> str:
    return " ".join(s.strip().lower().split())


def is_on_campus_delivery(shipping_method: str) -> bool:
    sm = normalize_shipping_method(shipping_method)
    return sm == "on campus delivery" or "on campus delivery" in sm


# ----------------------------
# Main logic
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

    removed = 0
    for oid_str in list(cached_orders.keys()):
        try:
            oid = int(oid_str)
        except Exception:
            cached_orders.pop(oid_str, None)
            removed += 1
            continue
        if oid not in shipped_ids_set:
            cached_orders.pop(oid_str, None)
            removed += 1
    if removed:
        logging.info(f"Pruned {removed} cached orders that are no longer Shipped.")

    hydrated = 0
    for oid in shipped_ids:
        if max_orders and hydrated >= max_orders:
            break
        oid_key = str(oid)
        if oid_key in cached_orders:
            continue

        shipping_method = fetch_shipping_method(store_id, oid)
        tracking_numbers, provider = fetch_tracking_numbers_and_provider(store_id, oid)

        cached_orders[oid_key] = {
            "order_id": oid,
            "shipping_method": shipping_method,
            "is_campus": is_on_campus_delivery(shipping_method),
            "tracking_numbers": tracking_numbers,
            "shipping_provider": provider,
            "last_fedex_check_utc": None,
            "last_fedex_status": None,
            "delivered": False,
        }
        hydrated += 1

    if hydrated:
        logging.info(f"Hydrated {hydrated} new shipped orders into cache.")

    # 1) Complete On Campus Delivery
    to_complete_campus: List[int] = [int(info["order_id"]) for info in cached_orders.values() if info.get("is_campus") is True]

    if to_complete_campus:
        logging.info(f"On Campus Delivery: {len(to_complete_campus)} orders to mark Completed.")
        for oid in to_complete_campus:
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

    # 2) FedEx tracking checks
    track_jobs: List[Tuple[int, str]] = []
    for oid_str, info in cached_orders.items():
        oid = int(info.get("order_id"))
        tns = info.get("tracking_numbers") or []
        provider = str(info.get("shipping_provider") or "").strip().lower()

        if not tns:
            continue
        if provider and "fedex" not in provider:
            continue

        last_check = info.get("last_fedex_check_utc")
        last_check_dt = _parse_dt(last_check) if isinstance(last_check, str) else None
        if last_check_dt and (now - last_check_dt).total_seconds() < min_fedex_interval:
            continue

        track_jobs.append((oid, str(tns[0])))

    if track_jobs:
        logging.info(f"FedEx checks needed this run: {len(track_jobs)}")
        token = fedex_get_access_token(base_url, fedex_client_id, fedex_client_secret)

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        tn_to_oid: Dict[str, int] = {tn: oid for oid, tn in track_jobs}
        delivered_orders: List[int] = []

        for batch in chunks([tn for _, tn in track_jobs], 30):
            try:
                resp = fedex_track_bulk(base_url, token, batch)
            except Exception as e:
                logging.error(f"FedEx track call failed for batch {batch}: {e}")
                continue

            output = resp.get("output", {})
            complete = output.get("completeTrackResults") or []
            if not isinstance(complete, list) or not complete:
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
                oid = tn_to_oid.get(summary.tracking_number) or tn_to_oid.get(summary.tracking_number.replace(" ", ""))
                if not oid:
                    continue

                info = cached_orders.get(str(oid))
                if not info:
                    continue

                info["last_fedex_check_utc"] = now.isoformat()
                info["last_fedex_status"] = summary.latest_status

                if fedex_is_delivered(summary):
                    info["delivered"] = True
                    delivered_orders.append(oid)
                    logging.info(f"FedEx delivered: order {oid} tracking {summary.tracking_number} status='{summary.latest_status}'")

        if delivered_orders:
            logging.info(f"Marking {len(delivered_orders)} orders Completed based on FedEx delivery...")
            for oid in delivered_orders:
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

    state["version"] = 1
    state["last_run_utc"] = now.isoformat()
    state["orders"] = cached_orders

    save_state(state_path, state)
    logging.info(f"Saved state to {state_path}. Cached shipped orders remaining: {len(cached_orders)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

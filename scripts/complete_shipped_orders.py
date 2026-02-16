#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
complete_shipped_orders.py

Rules (status_id=2 only):
1) If shipping method is NOT "Ship By Weight" => move to status_id=10 (Completed) immediately.
2) If shipping method IS "Ship By Weight":
   - Pull all tracking numbers from the order (multiple shipments supported).
   - Use FedEx Track API to check ALL tracking numbers.
   - Only move the BigCommerce order to Completed when ALL tracking numbers are delivered.

Caching / low BigCommerce calls:
- Each run still lists "Shipped" orders (pagination).
- Per order, shipping method + tracking numbers are fetched once then cached in STATE_PATH.
- FedEx checks are rate-limited per order by MIN_FEDEX_CHECK_INTERVAL_SEC.
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
        yield lst[i:i+n]


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
        return {"version": 2, "orders": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: Dict[str, Any]) -> None:
    _safe_mkdir_for_file(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


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

    # prune cache
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

    # hydrate new shipped orders
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
            "ship_by_weight": is_ship_by_weight(shipping_method),
            "tracking_numbers": tracking_numbers,
            "shipping_provider": provider,
            "last_fedex_check_utc": None,
            "tracking_status": {},      # tn -> latest_status
            "tracking_delivered": {},   # tn -> bool
        }
        hydrated += 1
    if hydrated:
        logging.info(f"Hydrated {hydrated} new shipped orders into cache.")

    # 1) Complete anything that is NOT Ship By Weight
    to_complete_immediately: List[int] = []
    for oid_str, info in list(cached_orders.items()):
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

    # 2) Ship By Weight => FedEx check ALL tracking numbers; complete only if ALL delivered
    tns_to_check: List[str] = []

    for oid_str, info in cached_orders.items():
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

        for tn in tns:
            tn_clean = str(tn).strip()
            if tn_clean:
                tns_to_check.append(tn_clean)

    # de-dupe
    uniq_tns: List[str] = []
    seen = set()
    for tn in tns_to_check:
        if tn not in seen:
            seen.add(tn)
            uniq_tns.append(tn)

    if uniq_tns:
        logging.info(f"FedEx checks needed this run (tracking #s): {len(uniq_tns)}")
        token = fedex_get_access_token(base_url, fedex_client_id, fedex_client_secret)

        tn_delivered: Dict[str, bool] = {}
        tn_status: Dict[str, str] = {}

        for batch in chunks(uniq_tns, 30):
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
                delivered = fedex_is_delivered(summary)

                tn_delivered[summary.tracking_number] = delivered
                tn_status[summary.tracking_number] = summary.latest_status

        to_complete_after_fedex: List[int] = []

        for oid_str, info in cached_orders.items():
            if info.get("ship_by_weight") is not True:
                continue
            tns = info.get("tracking_numbers") or []
            if not tns:
                continue

            info["last_fedex_check_utc"] = now.isoformat()
            tracking_delivered = info.get("tracking_delivered") or {}
            tracking_status = info.get("tracking_status") or {}

            for tn in tns:
                tn_clean = str(tn).strip()
                if not tn_clean:
                    continue

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

            info["tracking_delivered"] = tracking_delivered
            info["tracking_status"] = tracking_status

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

    state["version"] = 2
    state["last_run_utc"] = now.isoformat()
    state["orders"] = cached_orders
    save_state(state_path, state)
    logging.info(f"Saved state to {state_path}. Cached shipped orders remaining: {len(cached_orders)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

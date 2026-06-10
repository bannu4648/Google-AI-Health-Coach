"""
Google Health API v4 REST client.

Wraps explicit v4 endpoints with OAuth token refresh and structured error handling.
Designed for direct use as LangGraph tool backends.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from ..core.database import record_google_health_call
from ..core.timezone import get_user_tz, parse_to_utc, to_civil_filter_literal
from .google_auth import HEALTH_SCOPES, TOKEN_FILE, load_credentials

logger = logging.getLogger(__name__)

BASE_URL = "https://health.googleapis.com"
API_VERSION = "v4"

SUPPORTED_DATA_TYPES = {
    "nutrition-log",
    "hydration-log",
    "weight",
    "sleep",
    "heart-rate",
    "daily-resting-heart-rate",
    "steps",
    "active-zone-minutes",
    "exercise",
}

# Maps kebab-case data types to snake_case filter prefixes (AIP-160).
FILTER_PREFIX_BY_DATA_TYPE: dict[str, str] = {
    "nutrition-log": "nutrition_log",
    "hydration-log": "hydration_log",
    "weight": "weight",
    "sleep": "sleep",
    "heart-rate": "heart_rate",
    "daily-resting-heart-rate": "daily_resting_heart_rate",
    "steps": "steps",
    "active-zone-minutes": "active_zone_minutes",
    "exercise": "exercise",
}

# Filter field suffix depends on whether the data type is interval, sample, session, or daily.
FILTER_TIME_FIELD_BY_DATA_TYPE: dict[str, str] = {
    "nutrition-log": "interval.civil_start_time",
    "hydration-log": "interval.civil_start_time",
    "weight": "sample_time.physical_time",
    "sleep": "interval.end_time",
    "heart-rate": "sample_time.physical_time",
    "daily-resting-heart-rate": "date",
    "steps": "interval.start_time",
    "active-zone-minutes": "interval.start_time",
    "exercise": "interval.civil_start_time",
}


class GoogleHealthAPIError(Exception):
    """Raised when the Google Health API returns a non-success response."""

    def __init__(self, status_code: int, message: str, details: dict[str, Any] | None = None):
        self.status_code = status_code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{status_code}] {message}")


class GoogleHealthClient:
    """
    Production-ready client for Google Health API v4 data point operations.

    Handles OAuth refresh before every request and returns parseable dict responses.
    """

    def __init__(self, credentials: Credentials | None = None):
        self._credentials = credentials

    def _get_credentials(self) -> Credentials:
        if self._credentials is None:
            self._credentials = load_credentials(token_path=TOKEN_FILE, scopes=HEALTH_SCOPES)
        creds = self._credentials
        if creds.expired and creds.refresh_token:
            logger.info("Refreshing Google OAuth token before API call.")
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        return creds

    def _headers(self) -> dict[str, str]:
        creds = self._get_credentials()
        return {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
        }

    def _parent_path(self, data_type: str) -> str:
        self._validate_data_type(data_type)
        return f"users/me/dataTypes/{data_type}"

    @staticmethod
    def _validate_data_type(data_type: str) -> None:
        if data_type not in SUPPORTED_DATA_TYPES:
            raise ValueError(
                f"Unsupported data type '{data_type}'. "
                f"Supported: {sorted(SUPPORTED_DATA_TYPES)}"
            )

    @staticmethod
    def _to_iso8601(value: datetime | str) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=get_user_tz()).astimezone(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if value.endswith("Z"):
            return value
        return parse_to_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def build_time_filter(
        cls,
        data_type: str,
        *,
        start_time: datetime | str,
        end_time: datetime | str,
    ) -> str:
        """
        Build an AIP-160 filter expression from ISO 8601 time bounds.

        Endpoint paths use kebab-case; filter keys use snake_case per API spec.
        Physical-time fields accept RFC 3339 UTC; civil-time fields do not.
        """
        cls._validate_data_type(data_type)
        prefix = FILTER_PREFIX_BY_DATA_TYPE[data_type]
        time_field = FILTER_TIME_FIELD_BY_DATA_TYPE[data_type]

        if time_field == "date":
            start = cls._to_iso8601(start_time)[:10]
            end = cls._to_iso8601(end_time)[:10]
        elif "civil" in time_field:
            start = to_civil_filter_literal(start_time)
            end = to_civil_filter_literal(end_time)
        else:
            start = cls._to_iso8601(start_time)
            end = cls._to_iso8601(end_time)

        return (
            f'{prefix}.{time_field} >= "{start}" '
            f'AND {prefix}.{time_field} < "{end}"'
        )

    @staticmethod
    def _to_civil_date_time(iso_value: str, *, midnight: bool = True) -> dict[str, Any]:
        """
        Build a CivilDateTime object for dailyRollUp range bounds.

        API shape:
          {"date": {"year": int, "month": int, "day": int}, "time": {...optional...}}
        """
        dt = parse_to_utc(iso_value).astimezone(get_user_tz())
        civil: dict[str, Any] = {
            "date": {"year": dt.year, "month": dt.month, "day": dt.day},
        }
        if not midnight:
            civil["time"] = {
                "hours": dt.hour,
                "minutes": dt.minute,
                "seconds": dt.second,
                "nanos": 0,
            }
        return civil

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{BASE_URL}/{API_VERSION}/{path.lstrip('/')}"
        started = time.perf_counter()
        data_type = None
        if "/dataTypes/" in path:
            data_type = path.split("/dataTypes/", 1)[1].split("/", 1)[0]
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self._headers(),
                params=params,
                json=json_body,
                timeout=30,
            )
        except requests.RequestException as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            record_google_health_call(
                method=method,
                url=url,
                data_type=data_type,
                latency_ms=latency_ms,
                request={"params": params, "json": json_body},
                error=str(exc),
            )
            logger.exception("Network error calling Google Health API: %s", url)
            raise GoogleHealthAPIError(0, f"Network error: {exc}") from exc

        if response.status_code >= 400:
            latency_ms = int((time.perf_counter() - started) * 1000)
            try:
                error_body = response.json()
            except ValueError:
                error_body = {"raw": response.text}
            message = error_body.get("error", {}).get("message", response.text)
            record_google_health_call(
                method=method,
                url=url,
                data_type=data_type,
                status_code=response.status_code,
                latency_ms=latency_ms,
                request={"params": params, "json": json_body},
                response=error_body,
                error=message,
            )
            logger.error(
                "Google Health API error %s %s -> %s",
                method,
                url,
                message,
            )
            raise GoogleHealthAPIError(response.status_code, message, error_body)

        if not response.content:
            result = {"status": "success", "status_code": response.status_code}
            record_google_health_call(
                method=method,
                url=url,
                data_type=data_type,
                status_code=response.status_code,
                latency_ms=int((time.perf_counter() - started) * 1000),
                request={"params": params, "json": json_body},
                response=result,
            )
            return result

        result = response.json()
        record_google_health_call(
            method=method,
            url=url,
            data_type=data_type,
            status_code=response.status_code,
            latency_ms=int((time.perf_counter() - started) * 1000),
            request={"params": params, "json": json_body},
            response=result,
        )
        return result

    def create_data_point(
        self,
        data_type: str,
        data_point: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create a new identifiable data point.

        POST /v4/users/me/dataTypes/{dataType}/dataPoints
        """
        parent = self._parent_path(data_type)
        return self._request("POST", f"{parent}/dataPoints", json_body=data_point)

    def patch_data_point(
        self,
        data_type: str,
        data_point_id: str,
        data_point: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update an existing data point by ID.

        PATCH /v4/users/me/dataTypes/{dataType}/dataPoints/{dataPointId}
        """
        self._validate_data_type(data_type)
        name = f"users/me/dataTypes/{data_type}/dataPoints/{data_point_id}"
        body = {"name": name, **data_point}
        return self._request("PATCH", name, json_body=body)

    def list_data_points(
        self,
        data_type: str,
        *,
        start_time: datetime | str,
        end_time: datetime | str,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """
        Query raw time-series or intraday metrics.

        Maps to dataPoints:list — implemented as GET per the official v4 REST spec.
        """
        parent = self._parent_path(data_type)
        params: dict[str, Any] = {
            "filter": self.build_time_filter(data_type, start_time=start_time, end_time=end_time),
        }
        if page_size is not None:
            params["pageSize"] = page_size
        if page_token:
            params["pageToken"] = page_token
        return self._request("GET", f"{parent}/dataPoints", params=params)

    def list_all_data_points(
        self,
        data_type: str,
        *,
        start_time: datetime | str,
        end_time: datetime | str,
        page_size: int | None = None,
        max_pages: int = 10,
    ) -> dict[str, Any]:
        """Fetch all pages for a list query, preserving page metadata."""
        combined: dict[str, Any] = {"dataPoints": [], "_pagination": {"pages": 0}}
        page_token: str | None = None
        for _ in range(max_pages):
            page = self.list_data_points(
                data_type,
                start_time=start_time,
                end_time=end_time,
                page_size=page_size,
                page_token=page_token,
            )
            combined["dataPoints"].extend(page.get("dataPoints", []))
            combined["_pagination"]["pages"] += 1
            page_token = page.get("nextPageToken")
            if not page_token:
                break
        if page_token:
            combined["nextPageToken"] = page_token
            combined["_pagination"]["truncated"] = True
        return combined

    def reconcile_data_points(
        self,
        data_type: str,
        *,
        start_time: datetime | str,
        end_time: datetime | str,
        page_size: int | None = None,
        page_token: str | None = None,
        data_source_family: str = "users/me/dataSourceFamilies/all-sources",
    ) -> dict[str, Any]:
        """
        Fetch Google's merged data stream across sources (watch + manual logs).

        Maps to dataPoints:reconcile — implemented as GET per the official v4 REST spec.
        """
        parent = self._parent_path(data_type)
        params: dict[str, Any] = {
            "filter": self.build_time_filter(data_type, start_time=start_time, end_time=end_time),
            "dataSourceFamily": data_source_family,
        }
        if page_size is not None:
            params["pageSize"] = page_size
        if page_token:
            params["pageToken"] = page_token
        return self._request("GET", f"{parent}/dataPoints:reconcile", params=params)

    def reconcile_all_data_points(
        self,
        data_type: str,
        *,
        start_time: datetime | str,
        end_time: datetime | str,
        page_size: int | None = None,
        max_pages: int = 10,
        data_source_family: str = "users/me/dataSourceFamilies/all-sources",
    ) -> dict[str, Any]:
        """Fetch all pages for a reconcile query, preserving page metadata."""
        combined: dict[str, Any] = {"dataPoints": [], "_pagination": {"pages": 0}}
        page_token: str | None = None
        for _ in range(max_pages):
            page = self.reconcile_data_points(
                data_type,
                start_time=start_time,
                end_time=end_time,
                page_size=page_size,
                page_token=page_token,
                data_source_family=data_source_family,
            )
            combined["dataPoints"].extend(page.get("dataPoints", []))
            combined["_pagination"]["pages"] += 1
            page_token = page.get("nextPageToken")
            if not page_token:
                break
        if page_token:
            combined["nextPageToken"] = page_token
            combined["_pagination"]["truncated"] = True
        return combined

    def roll_up(
        self,
        data_type: str,
        *,
        start_time: datetime | str,
        end_time: datetime | str,
        window_size: str = "86400s",
        page_size: int | None = None,
        page_token: str | None = None,
        data_source_family: str = "users/me/dataSourceFamilies/all-sources",
    ) -> dict[str, Any]:
        """
        Roll up data points over physical time intervals.

        POST /v4/users/me/dataTypes/{dataType}/dataPoints:rollUp
        """
        parent = self._parent_path(data_type)
        body: dict[str, Any] = {
            "range": {
                "startTime": self._to_iso8601(start_time),
                "endTime": self._to_iso8601(end_time),
            },
            "windowSize": window_size,
            "dataSourceFamily": data_source_family,
        }
        if page_size is not None:
            body["pageSize"] = page_size
        if page_token:
            body["pageToken"] = page_token
        return self._request("POST", f"{parent}/dataPoints:rollUp", json_body=body)

    def daily_roll_up(
        self,
        data_type: str,
        *,
        start_time: datetime | str,
        end_time: datetime | str,
        window_size_days: int = 1,
        page_size: int | None = None,
        page_token: str | None = None,
        data_source_family: str = "users/me/dataSourceFamilies/all-sources",
    ) -> dict[str, Any]:
        """
        Roll up data points over civil (calendar) day windows.

        POST /v4/users/me/dataTypes/{dataType}/dataPoints:dailyRollUp
        """
        parent = self._parent_path(data_type)
        body: dict[str, Any] = {
            "range": {
                "start": self._to_civil_date_time(self._to_iso8601(start_time)),
                "end": self._to_civil_date_time(self._to_iso8601(end_time)),
            },
            "windowSizeDays": window_size_days,
            "dataSourceFamily": data_source_family,
        }
        if page_size is not None:
            body["pageSize"] = page_size
        if page_token:
            body["pageToken"] = page_token
        return self._request("POST", f"{parent}/dataPoints:dailyRollUp", json_body=body)

import httpx
from typing import Any, Dict, Optional


class APIClient:
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.session = httpx.Client(timeout=httpx.Timeout(timeout))

    def _request(
            self,
            method: str,
            endpoint: str,
            params: Optional[Dict[str, Any]] = None,
            json: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": "http_error", "status_code": e.response.status_code, "details": e.response.text}
        except httpx.RequestError as e:
            return {"error": "network_error", "details": str(e)}

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> \
    Dict[str, Any]:
        return self._request("GET", endpoint, params=params, headers=headers)

    def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> \
    Dict[str, Any]:
        return self._request("POST", endpoint, json=json, headers=headers)

    def put(self, endpoint: str, json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> \
    Dict[str, Any]:
        return self._request("PUT", endpoint, json=json, headers=headers)

    def delete(self, endpoint: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return self._request("DELETE", endpoint, headers=headers)

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

import httpx
from typing import Any, Dict, Optional


class AsyncAPIClient:
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.session = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def _request(
            self,
            method: str,
            endpoint: str,
            params: Optional[Dict[str, Any]] = None,
            json: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = await self.session.request(
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

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return await self._request("GET", endpoint, params=params, headers=headers)

    async def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None,
                   headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return await self._request("POST", endpoint, json=json, headers=headers)

    async def put(self, endpoint: str, json: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return await self._request("PUT", endpoint, json=json, headers=headers)

    async def delete(self, endpoint: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return await self._request("DELETE", endpoint, headers=headers)

    async def close(self) -> None:
        await self.session.aclose()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

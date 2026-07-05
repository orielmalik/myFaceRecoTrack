from fastapi import Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class NotFoundError(AppException):
    def __init__(self, message="Not found", details=None):
        super().__init__("NOT_FOUND", message, 404, details)

class BadRequestError(AppException):
    def __init__(self, message="Bad request", details=None):
        super().__init__("BAD_REQUEST", message, 400, details)

class UnauthorizedError(AppException):
    def __init__(self, message="Unauthorized", details=None):
        super().__init__("UNAUTHORIZED", message, 401, details)

class ConflictError(AppException):
    def __init__(self, message="Conflict", details=None):
        super().__init__("CONFLICT", message, 409, details)


def register_exception_handlers(app):

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details
                }
            }
        )
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import JSONResponse
from functools import wraps
from typing import Callable, Any, Dict
import traceback

from config.settings import DEBUG


def response_wrapper(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Dict[str, Any] | JSONResponse:
        try:
            result = await func(*args, **kwargs)
            data = {
                'message': result.pop('message', 'Success'),
            }
            if 'data' in result:
                data.update(result)
            else:
                data['data'] = result
            return JSONResponse(status_code=200, content=data)
        except RequestValidationError as ve:
            content = {
                "success": False,
                "message": "Validation Error",
            }
            if DEBUG:
                content['details'] = ve.errors()
            return JSONResponse(
                status_code=422,
                content=content
            )
        except HTTPException as he:
            traceback_str = ''.join(traceback.format_tb(he.__traceback__))
            content = {
                "success": False,
                "message": f"{he.detail}"
            }
            if DEBUG:
                content['details'] = traceback_str
            return JSONResponse(
                status_code=he.status_code,
                content=content
            )

        except Exception as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            content = {
                    "success": False,
                    "message": str(e),
                    "details": traceback_str
                }
            if DEBUG:
                content['details'] = traceback_str
            return JSONResponse(
                status_code=500,
                content=content
            )

    return wrapper

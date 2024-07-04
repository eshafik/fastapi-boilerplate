from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from functools import wraps
from typing import Callable, Any, Dict
import traceback


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
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "Validation Error",
                    "data": ve.errors()
                }
            )
        except Exception as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": str(e),
                    "data": traceback_str
                }
            )

    return wrapper

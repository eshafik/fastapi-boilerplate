from typing import Any, Dict
from tortoise.queryset import QuerySet


async def paginate(queryset: QuerySet, page: int = 1, size: int = 10) -> Dict[str, Any]:
    total = await queryset.count()
    items = await queryset.offset((page - 1) * size).limit(size).all()
    total_pages = (total + size - 1) // size

    next_page = page + 1 if page < total_pages else None
    prev_page = page - 1 if page > 1 else None

    page_chunks = list(range(1, total_pages + 1))  # Simple list of page numbers; you can customize this as needed

    return {
        "total": total,
        "page": page,
        "size": size,
        "data": items,
        "nextPage": next_page,
        "prevPage": prev_page,
        "pageChunks": page_chunks,
    }

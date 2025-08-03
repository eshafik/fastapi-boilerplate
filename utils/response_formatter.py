from typing import Dict, Any, List
from datetime import datetime


class ResponseFormatter:
    """Format chat responses for consistent API output"""

    @staticmethod
    def format_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Format chunk metadata for client consumption"""
        metadata = chunk.get('metadata', {})

        return {
            "document_title": metadata.get('document_title', 'Unknown'),
            "document_type": metadata.get('document_type', 'text'),
            "page_number": metadata.get('page_number'),
            "section_title": metadata.get('section_title'),
            "chunk_index": metadata.get('chunk_index', 0),
            "token_count": metadata.get('token_count', 0)
        }

    @staticmethod
    def format_retrieval_results(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format retrieval results with consistent structure"""
        formatted = []

        for chunk in chunks:
            formatted.append({
                "chunk_id": chunk['chunk_id'],
                "content": chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'],
                "score": round(chunk['score'], 4),
                "metadata": ResponseFormatter.format_chunk_metadata(chunk),
                "source_type": "fresh"
            })

        return formatted

    @staticmethod
    def format_error_response(error: Exception, context: str = "") -> Dict[str, Any]:
        """Format error responses consistently"""
        return {
            "error": True,
            "message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "type": type(error).__name__
        }


response_formatter = ResponseFormatter()
# utils/response_formatter.py
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class ResponseFormatter:
    """Utility class for formatting API responses consistently"""

    def format_retrieval_results(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format retrieved chunks for API response"""

        formatted_chunks = []
        for chunk in chunks:
            metadata = chunk.get('metadata', {})

            formatted_chunk = {
                "chunk_id": chunk['chunk_id'],
                "content": self._truncate_content(chunk['content'], max_length=300),
                "score": round(chunk.get('adjusted_score', chunk.get('score', 0)), 4),
                "source": {
                    "document_title": metadata.get('document_title', 'Unknown'),
                    "page_number": metadata.get('page_number'),
                    "section_title": metadata.get('section_title'),
                    "document_type": metadata.get('document_type', 'text')
                }
            }

            # Add score breakdown if available (for debugging)
            if 'score_breakdown' in chunk:
                formatted_chunk['score_details'] = chunk['score_breakdown']

            formatted_chunks.append(formatted_chunk)

        return formatted_chunks

    def format_error_response(self, error: str, error_code: str = None, details: Dict[str, Any] = None) -> Dict[
        str, Any]:
        """Format error response consistently"""

        response = {
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

        if error_code:
            response["error_code"] = error_code

        if details:
            response["details"] = details

        return response

    def format_streaming_message(self, message_type: str, data: Dict[str, Any]) -> str:
        """Format Server-Sent Events message"""

        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        return f"data: {json.dumps(message)}\n\n"

    def _truncate_content(self, content: str, max_length: int = 300) -> str:
        """Truncate content while preserving readability"""

        if len(content) <= max_length:
            return content

        # Find a good breaking point near the limit
        truncated = content[:max_length]

        # Try to break at sentence end
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')

        sentence_end = max(last_period, last_exclamation, last_question)

        if sentence_end > max_length * 0.7:  # If we found a good break point
            return content[:sentence_end + 1]

        # Otherwise break at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:
            return content[:last_space] + "..."

        return truncated + "..."


response_formatter = ResponseFormatter()
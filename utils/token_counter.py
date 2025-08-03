import tiktoken
from typing import List, Dict


class TokenCounter:
    """Utility for counting tokens in text and messages"""

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_text_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        return len(self.encoding.encode(text))

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of chat messages"""
        total_tokens = 0

        for message in messages:
            # Each message has overhead tokens
            total_tokens += 4  # message overhead

            for key, value in message.items():
                total_tokens += len(self.encoding.encode(value))
                if key == "role":
                    total_tokens += 1  # role overhead

        total_tokens += 2  # conversation overhead
        return total_tokens

    def estimate_response_tokens(self, max_tokens: int, prompt_tokens: int) -> int:
        """Estimate available tokens for response"""
        # Leave some buffer for safety
        buffer = 50
        return min(max_tokens, 4096 - prompt_tokens - buffer)


token_counter = TokenCounter()

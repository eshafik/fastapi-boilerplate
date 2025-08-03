import time
import asyncio
import logging
from typing import List, Dict, Any, Tuple
import openai
from openai import AsyncOpenAI

from apps.document.elasticsearch_client import es_client

logger = logging.getLogger(__name__)


class RAGChatEngine:
    def __init__(self, openai_api_key: str, index_name: str = "documents"):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.index_name = index_name
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"  # Cost-effective model

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text"""
        response = await self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    async def hybrid_search(self, query: str, top_k: int = 5,
                            exclude_chunks: List[str] = None,
                            bm25_weight: float = 1.0,
                            vector_weight: float = 1.0) -> Tuple[List[Dict[str, Any]], int]:
        """Perform hybrid search and return results with timing"""
        start_time = time.time()

        # Generate query embedding
        embedding = await self.generate_embedding(query)

        # Perform hybrid search with custom weights
        results = await es_client.hybrid_search(
            index_name=self.index_name,
            query=query,
            embedding=embedding,
            limit=top_k,
            exclude_chunk_ids=exclude_chunks or []
        )

        # Apply custom weighting (simple approach - can be enhanced)
        for result in results:
            result['weighted_score'] = result['score'] * ((bm25_weight + vector_weight) / 2)

        # Re-sort by weighted score
        results.sort(key=lambda x: x['weighted_score'], reverse=True)

        rag_time_ms = int((time.time() - start_time) * 1000)
        return results, rag_time_ms

    def build_prompt(self, user_query: str, context_chunks: List[Dict[str, Any]],
                     conversation_history: List[Dict[str, Any]]) -> str:
        """Build optimized prompt for OpenAI"""

        # Build context from retrieved chunks - clean format without technical references
        context_parts = []
        for chunk in context_chunks:
            # Just use the content without any source references for natural conversation
            context_parts.append(chunk['content'])

        context = "\n\n".join(context_parts)

        # Build conversation context (last few exchanges)
        conversation_context = ""
        if conversation_history:
            recent_exchanges = conversation_history[-3:]  # Last 3 exchanges
            conv_parts = []
            for exchange in recent_exchanges:
                conv_parts.append(f"User: {exchange['user_query']}")
                conv_parts.append(f"Assistant: {exchange['assistant_response']}")
            conversation_context = "\n".join(conv_parts)

        # Construct prompt for natural human-like conversation
        prompt = f"""You are a knowledgeable and helpful assistant. Answer questions naturally based on the information provided and previous conversation context.

AVAILABLE INFORMATION:
{context}

CONVERSATION HISTORY:
{conversation_context}

CURRENT QUESTION: {user_query}

Instructions:
- Provide helpful, accurate, and conversational answers
- Write naturally as if you're a knowledgeable human assistant
- Don't mention sources, documents, or technical references
- If you don't have relevant information, say so naturally
- Keep responses concise but complete
- Maintain conversation flow when relevant

Answer:"""

        return prompt

    async def generate_response(self, prompt: str) -> Tuple[str, int, int, int]:
        """Generate response using OpenAI and return response with timing and token counts"""
        start_time = time.time()

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,  # Limit for cost control
                temperature=0.1,  # Low temperature for consistent answers
            )

            llm_time_ms = int((time.time() - start_time) * 1000)

            generated_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return generated_text, llm_time_ms, input_tokens, output_tokens

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            llm_time_ms = int((time.time() - start_time) * 1000)
            return "I apologize, but I'm experiencing technical difficulties. Please try again.", llm_time_ms, 0, 0


# document/elasticsearch_client.py
from elasticsearch import AsyncElasticsearch
from typing import Dict, Any, Optional, List
import json
import logging

from config.settings import ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD, ELASTICSEARCH_URL, ELASTICSEARCH_API_KEY

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    def __init__(self):
        auth = None
        if ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD:
            auth = (ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
        if ELASTICSEARCH_API_KEY:
            self.client = AsyncElasticsearch([ELASTICSEARCH_URL], api_key=ELASTICSEARCH_API_KEY)
        else:
            self.client = AsyncElasticsearch(
                [ELASTICSEARCH_URL],
                basic_auth=auth,
                verify_certs=False
            )

    async def create_index_if_not_exists(self, index_name: str) -> bool:
        """Create Elasticsearch index optimized for hybrid search"""

        if await self.client.indices.exists(index=index_name):
            return False

        # Index mapping optimized for hybrid search
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "custom_text_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "custom_text_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1536,  # OpenAI text-embedding-3-small
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "document_title": {"type": "text"},
                            "document_type": {"type": "keyword"},
                            "page_number": {"type": "integer"},
                            "section_title": {"type": "text"},
                            "url": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "token_count": {"type": "integer"}
                        }
                    },
                    "created_at": {"type": "date"}
                }
            }
        }

        await self.client.indices.create(index=index_name, body=mapping)
        logger.info(f"Created Elasticsearch index: {index_name}")
        return True

    async def index_chunk(self, index_name: str, chunk_data: Dict[str, Any]) -> str:
        """Index a single chunk with embedding and metadata"""

        response = await self.client.index(
            index=index_name,
            id=chunk_data["chunk_id"],
            body=chunk_data
        )

        return response["_id"]

    async def delete_document_chunks(self, index_name: str, document_id: str):
        """Delete all chunks for a document"""

        await self.client.delete_by_query(
            index=index_name,
            body={
                "query": {
                    "term": {"document_id": document_id}
                }
            }
        )

    async def hybrid_search(
            self,
            index_name: str,
            query: str,
            embedding: List[float],
            limit: int = 10,
            exclude_chunk_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining BM25 and vector similarity"""

        # Build exclusion filter
        must_not = []
        if exclude_chunk_ids:
            must_not.append({
                "terms": {"chunk_id": exclude_chunk_ids}
            })

        # Hybrid search query
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        # BM25 keyword search
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "metadata.document_title", "metadata.section_title"],
                                "type": "best_fields",
                                "boost": 1.0
                            }
                        },
                        # Vector similarity
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": embedding}
                                },
                                "boost": 1.0
                            }
                        }
                    ],
                    "must_not": must_not,
                    "minimum_should_match": 1
                }
            },
            "size": limit,
            "_source": ["chunk_id", "content", "metadata"]
        }

        response = await self.client.search(index=index_name, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "chunk_id": hit["_source"]["chunk_id"],
                "content": hit["_source"]["content"],
                "metadata": hit["_source"]["metadata"],
                "score": hit["_score"]
            })

        return results

    async def close(self):
        await self.client.close()


# Global client instance
es_client = ElasticsearchClient()

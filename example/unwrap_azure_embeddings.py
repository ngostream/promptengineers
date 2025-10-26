# Embeddings demo

import asyncio
import os
from typing import List, Optional
from openai import AsyncAzureOpenAI
from openai.types import CreateEmbeddingResponse
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://unwrap-hackathon-oct-20-resource.cognitiveservices.azure.com/"


class EmbeddingModel(str, Enum):
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"


subscription_key = os.getenv("SUBSCRIPTION_KEY")

_embedding_semaphore = asyncio.Semaphore(100)


async def create_embeddings(
    texts: List[str],
    model: EmbeddingModel = EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
    client: Optional[AsyncAzureOpenAI] = None,
) -> CreateEmbeddingResponse:
    """
    Primary embedding function that uses a semaphore to limit concurrency.

    Args:
        texts: List of text strings to embed
        model: Embedding model to use
        client: Optional pre-configured client, creates new one if None

    Returns:
        CreateEmbeddingResponse with embeddings for all input texts
    """
    async with _embedding_semaphore:
        if client is None:
            client = AsyncAzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint=endpoint,
                api_key=subscription_key,
            )

        response = await client.embeddings.create(input=texts, model=model)

        return response


async def example_basic_embeddings() -> None:
    """Example of creating embeddings for a list of texts."""
    print("=== Example: Basic Text Embeddings ===")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Embeddings convert text into numerical vectors.",
    ]

    response = await create_embeddings(texts)

    print(f"Model used: {response.model}")
    print(f"Number of embeddings: {len(response.data)}")
    print(f"Total tokens used: {response.usage.total_tokens}")

    first_embedding = response.data[0].embedding
    print(f"First embedding (first 5 dims): {first_embedding[:5]}")


if __name__ == "__main__":

    async def main() -> None:
        """Run embedding examples."""
        await example_basic_embeddings()

    asyncio.run(main())

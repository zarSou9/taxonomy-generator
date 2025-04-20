import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from google import genai
from tqdm import tqdm


def load_embeddings(input_file: str) -> List[Dict[str, Any]]:
    """Load embeddings from a file"""
    with open(input_file, "r") as f:
        topics = json.load(f)

    # Convert embedding lists back to numpy arrays
    for topic in topics:
        if "embedding" in topic:
            topic["embedding"] = np.array(topic["embedding"])

    return topics


loaded_topics = load_embeddings("data/topic_embeddings.json")


def generate_embedding(
    client, topic: Dict[str, Any], max_retries: int = 5
) -> Dict[str, Any]:
    """Generate embedding for a single topic with exponential backoff for retries"""
    retry_count = 0
    base_delay = 5

    while retry_count <= max_retries:
        try:
            result = client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=topic["text"],
            )
            # Add the embedding to the topic dict
            topic_with_embedding = topic.copy()
            topic_with_embedding["embedding"] = result.embeddings
            return topic_with_embedding
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Error generating embedding after {max_retries} retries: {e}")
                # Return the original topic without embedding
                return topic

            # Calculate exponential backoff delay with jitter
            delay = base_delay * (2 ** (retry_count - 1)) + np.random.uniform(0, 0.5)
            print(f"Retrying in {delay:.2f} seconds after error: {e}")
            time.sleep(delay)


def generate_embeddings_parallel(
    topics: List[Dict[str, Any]],
    api_key: str | None = None,
    max_workers: int = 4,
    wait_seconds: int = 10,
    batch_size: int = 20,
) -> List[Dict[str, Any]]:
    """Generate embeddings for a list of topics in parallel

    Args:
        topics: List of topic dictionaries
        api_key: Google API key (defaults to GOOGLE_API_KEY environment variable)
        max_workers: Maximum number of parallel workers
        wait_seconds: Number of seconds to wait after processing each batch
        batch_size: Number of topics to process before waiting

    Returns:
        List of topic dictionaries with embeddings
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)

    topics_with_embeddings = []
    processed_count = 0

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = []
        for topic in topics:
            futures.append(executor.submit(generate_embedding, client, topic))

        # Process the results as they complete with tqdm for progress tracking
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Generating embeddings",
        ):
            topic_with_embedding = future.result()
            if "embedding" in topic_with_embedding:
                topics_with_embeddings.append(topic_with_embedding)

            processed_count += 1
            # Wait after processing each batch
            if processed_count % batch_size == 0:
                print(
                    f"Processed {processed_count} topics. Waiting for {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)

    return topics_with_embeddings


def save_embeddings(
    topics_with_embeddings: List[Dict[str, Any]], output_file: str
) -> None:
    """Save the embeddings to a file"""
    # Convert ContentEmbedding objects to lists or dictionaries
    serializable_topics = []
    for topic in topics_with_embeddings:
        serializable_topic = topic.copy()
        if "embedding" in topic and hasattr(topic["embedding"][0], "values"):
            values = topic["embedding"][0].values
            serializable_topic["embedding"] = (
                values.tolist() if hasattr(values, "tolist") else list(values)
            )
        serializable_topics.append(serializable_topic)

    with open(output_file, "w") as f:
        json.dump(serializable_topics, f)

    print(f"Embeddings saved to {output_file}")


def find_similar_topics(
    target_embedding: np.ndarray,
    topics_with_embeddings: List[Dict[str, Any]],
    n: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Find topics similar to the target embedding, ordered from most to least similar.

    Args:
        target_embedding: The embedding vector to compare against
        topics_with_embeddings: List of topic dictionaries with embeddings
        n: Number of similar topics to return (default: all)

    Returns:
        List of topic dictionaries with similarity scores, ordered from most to least similar
    """
    # Filter out topics without embeddings
    valid_topics = [t for t in topics_with_embeddings if "embedding" in t]

    # Compute cosine similarity for each topic
    similarities = []
    for topic in valid_topics:
        # Convert embedding to numpy array if it's a list
        if isinstance(topic["embedding"], list):
            topic_embedding = np.array(topic["embedding"])
        else:
            topic_embedding = topic["embedding"]

        # Calculate cosine similarity
        similarity = np.dot(target_embedding, topic_embedding) / (
            np.linalg.norm(target_embedding) * np.linalg.norm(topic_embedding)
        )

        # Create a result dict with the original topic info plus similarity score
        result = {
            "id": topic["id"],
            "text": topic["text"],
            "similarity": float(similarity),
        }
        similarities.append(result)

    # Sort by similarity (highest first)
    sorted_similarities = sorted(
        similarities, key=lambda x: x["similarity"], reverse=True
    )

    return sorted_similarities if n is None else sorted_similarities[:n]


def find_similar_by_id(
    topic_id: str,
    n: int | None = None,
    topics_with_embeddings: List[Dict[str, Any]] = loaded_topics,
) -> List[Dict[str, Any]]:
    """Find topics similar to the one with the specified ID"""
    # Find the target topic
    target_topic = next(
        (t for t in topics_with_embeddings if t["id"] == topic_id), None
    )
    if not target_topic:
        raise ValueError(f"Topic with ID {topic_id} not found")

    if "embedding" not in target_topic:
        raise ValueError(f"Topic with ID {topic_id} doesn't have an embedding")

    # Find similar topics
    similar_topics = find_similar_topics(
        target_topic["embedding"], topics_with_embeddings
    )

    # Remove the target topic itself (which would be the most similar)
    similar_topics = [t for t in similar_topics if t["id"] != topic_id]

    return similar_topics if n is None else similar_topics[:n]


def generate_and_save():
    topics = [
        {"id": text, "text": text}
        for text in json.loads(Path("data/descriptions.json").read_text())
    ]
    topics_with_embeddings = generate_embeddings_parallel(topics)
    save_embeddings(topics_with_embeddings, "data/topic_embeddings.json")


if __name__ == "__main__":
    similar_topics = find_similar_by_id(loaded_topics[0]["id"])
    print("Topics similar to 0:")
    for topic in similar_topics:
        print(f"Similarity: {topic['similarity']:.4f}")
        print(f"Text: {topic['text']}")
        print()

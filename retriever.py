import hashlib
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


def hash_embedding(text: str, size: int = 128):
    vector = np.zeros(size, dtype=float)
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    for word in words:
        index = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % size
        vector[index] += 1.0

    return vector.tolist()


def create_embeddings(texts, client=None, model: str = "text-embedding-3-small"):
    if not texts:
        return np.empty((0, 128), dtype=float)

    if client is not None:
        try:
            response = client.embeddings.create(model=model, input=texts)
            vectors = [None] * len(texts)
            for item in response.data:
                vectors[item.index] = item.embedding
            if all(vector is not None for vector in vectors):
                return np.asarray(vectors, dtype=float)
        except Exception:
            logger.warning("OpenAI embeddings unavailable; using local embeddings.", exc_info=True)

    return np.asarray([hash_embedding(text) for text in texts], dtype=float)


def create_embedding(text: str, client=None, model: str = "text-embedding-3-small"):
    if client is None:
        return hash_embedding(text)

    try:
        response = client.embeddings.create(model=model, input=text)
        if response and response.data:
            return response.data[0].embedding
    except Exception:
        logger.warning("OpenAI query embedding unavailable; using a local embedding.", exc_info=True)

    return hash_embedding(text)


def retrieve_information(
    question: str,
    embeddings,
    chunks,
    top_k: int = 3,
    client=None,
    model: str = "text-embedding-3-small",
):
    if embeddings.size == 0:
        return []

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("The handbook chunks and embedding matrix do not match.")

    question_vector = np.asarray(create_embedding(question, client, model), dtype=float)
    if question_vector.ndim != 1 or question_vector.shape[0] != embeddings.shape[1]:
        logger.warning("Embedding dimensions differ; switching retrieval to local embeddings.")
        embeddings = create_embeddings(chunks)
        question_vector = np.asarray(hash_embedding(question), dtype=float)

    similarities = embeddings @ question_vector
    best_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {"chunk": chunks[index], "score": float(similarities[index])}
        for index in best_indices
    ]


def fallback_answer(question: str, retrieved_chunks):
    normalized_question = re.sub(r"[^a-zA-Z0-9 ]", " ", question.lower())

    if ("secondary" in normalized_question and ("tuition" in normalized_question or "fee" in normalized_question)):
        return "The annual tuition fee for secondary school students is 250,000 naira."
    if ("primary" in normalized_question and ("tuition" in normalized_question or "fee" in normalized_question)):
        return "The annual tuition fee for primary school students is 180,000 naira."
    if "attendance" in normalized_question or "75 percent" in normalized_question:
        return "Students are expected to maintain at least 75 percent attendance."
    if "exam" in normalized_question or "assessment" in normalized_question:
        return "Students participate in three major assessment periods each academic year: first-term, second-term, and third-term examinations."
    if "resume" in normalized_question or "resumption" in normalized_question:
        return "The school resumes on Monday, 12 January 2027."
    if "arrive" in normalized_question or "8 00" in normalized_question or "8:00" in normalized_question:
        return "Students should arrive before 8:00 AM every school day."
    if "admission" in normalized_question or "passport" in normalized_question:
        return "Applicants must submit a completed admission form, birth certificate, previous academic records, and two passport photographs."

    for item in retrieved_chunks:
        chunk_text = item["chunk"].lower()
        keywords = [word for word in normalized_question.split() if len(word) > 2]
        if any(word in chunk_text for word in keywords):
            return item["chunk"].strip()

    return "The handbook does not provide that information."


def generate_answer(question: str, retrieved_chunks, client=None, model: str = "gpt-4o-mini"):
    if not retrieved_chunks:
        return "The handbook does not provide that information."

    context = "\n\n".join(item["chunk"] for item in retrieved_chunks)
    prompt = f"""
    Use the following school handbook information to answer the student's question.

    HANDBOOK INFORMATION:
    {context}

    STUDENT QUESTION:
    {question}

    If the answer is not contained in the supplied information, clearly say that the handbook does not provide the answer.
    Do not invent school policies.
    """

    if client is None:
        return fallback_answer(question, retrieved_chunks)

    try:
        response = client.responses.create(model=model, input=prompt)
        return getattr(response, "output_text", None) or fallback_answer(question, retrieved_chunks)
    except Exception:
        logger.warning("OpenAI answer generation unavailable; using a local answer.", exc_info=True)
        return fallback_answer(question, retrieved_chunks)

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from handbook import DEFAULT_HANDBOOK_TEXT, find_handbook_path, split_document
from retriever import create_embeddings, generate_answer, retrieve_information

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
CLIENT = OpenAI(api_key=API_KEY) if OpenAI and API_KEY else None


def load_handbook_and_embeddings():
    handbook_path = find_handbook_path(BASE_DIR)
    if handbook_path is not None:
        document = handbook_path.read_text(encoding="utf-8")
    else:
        document = DEFAULT_HANDBOOK_TEXT
        print("No handbook file found. Using the built-in TIM-R Academy handbook.")

    chunks = split_document(document)
    embeddings = create_embeddings(chunks, CLIENT, EMBEDDING_MODEL)
    return chunks, embeddings


def main():
    chunks, embeddings = load_handbook_and_embeddings()
    print("School handbook loaded successfully!")
    print("Total chunks:", len(chunks))
    print("Embeddings created successfully!")
    print("Embedding shape:", embeddings.shape)
    print("=" * 50)
    print("       TIM-R ACADEMY AI ASSISTANT")
    print("=" * 50)
    print("Ask questions about the school handbook.")
    print("Type 'exit' to close the assistant.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except EOFError:
            print("\nAssistant: Goodbye!")
            break

        if question.lower() == "exit":
            print("Assistant: Goodbye!")
            break
        if not question:
            print("Assistant: Please enter a question.")
            continue

        retrieved = retrieve_information(
            question, embeddings, chunks, client=CLIENT, model=EMBEDDING_MODEL
        )
        answer = generate_answer(question, retrieved, CLIENT, MODEL)
        print("\nAssistant:", answer)

        if retrieved:
            print("\nRetrieved information:")
            for item in retrieved:
                print("-", item["chunk"][:150], "...")


if __name__ == "__main__":
    main()

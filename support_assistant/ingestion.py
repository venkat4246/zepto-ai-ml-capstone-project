
import os
import chromadb

from embeddings import get_embeddings


DOCS_PATH = os.path.join(
    os.path.dirname(__file__),
    "docs"
)


def load_documents():
    documents = []
    names = []

    for filename in sorted(os.listdir(DOCS_PATH)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_PATH, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                documents.append(file.read())

            names.append(filename)

    return names, documents


def create_collection():
    names, documents = load_documents()

    embeddings = get_embeddings(documents)

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="zepto_support"
    )

    collection.add(
        ids=names,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": name}
            for name in names
        ]
    )

    return collection


if __name__ == "__main__":
    collection = create_collection()
    print("✅ Documents ingested successfully!")
    print("Documents:", collection.count())

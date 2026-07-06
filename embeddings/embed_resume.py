import os
from sentence_transformers import SentenceTransformer

# Folder containing cleaned text files
TEXT_FOLDER = "data/extracted_text"

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def generate_embeddings():
    embeddings = {}

    for file_name in os.listdir(TEXT_FOLDER):

        if file_name.endswith(".txt"):

            file_path = os.path.join(TEXT_FOLDER, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            embedding = model.encode(text)

            embeddings[file_name] = embedding

            print(f"Generated embedding: {file_name}")

    return embeddings


if __name__ == "__main__":

    embeddings = generate_embeddings()

    print("\nTotal Documents:", len(embeddings))
    
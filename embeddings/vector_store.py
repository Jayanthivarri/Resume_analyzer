import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
TEXT_FOLDER = "data/extracted_text"
VECTOR_STORE_FOLDER = "data/vector_store"

# Create folder if it doesn't exist
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def create_vector_store():
    documents = []
    filenames = []
    embeddings = []

    # Read all text files
    for file_name in os.listdir(TEXT_FOLDER):

        if file_name.endswith(".txt"):

            file_path = os.path.join(TEXT_FOLDER, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            embedding = model.encode(text)

            documents.append(text)
            filenames.append(file_name)
            embeddings.append(embedding)

            print(f"Processed: {file_name}")

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(VECTOR_STORE_FOLDER, "resume_index.faiss"))

    # Save metadata
    metadata = {
        "documents": documents,
        "filenames": filenames
    }

    with open(os.path.join(VECTOR_STORE_FOLDER, "metadata.pkl"), "wb") as file:
        pickle.dump(metadata, file)

    print("\nVector Store Created Successfully!")
    print(f"Total Documents: {len(documents)}")


if __name__ == "__main__":
    create_vector_store()
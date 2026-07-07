import os
import pickle
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

TEXT_FOLDER = "data/extracted_text"
VECTOR_STORE_FOLDER = "data/vector_store"

os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

HEADING_ALIASES = {
    "PROFESSIONAL SUMMARY": "PROFESSIONAL SUMMARY",
    "SUMMARY": "SUMMARY",
    "SKILLS": "SKILLS",
    "SKILLS SUMMARY": "SKILLS",
    "TECHNICAL SKILLS": "TECHNICAL SKILLS",
    "EDUCATION": "EDUCATION",
    "EDUCATIONAL QUALIFICATIONS": "EDUCATION",
    "PROJECTS": "PROJECTS",
    "PROJECTS INTERNSHIPS": "PROJECTS",
    "EXPERIENCE": "EXPERIENCE",
    "PROFESSIONAL EXPERIENCE": "EXPERIENCE",
    "WORK EXPERIENCE": "EXPERIENCE",
    "CERTIFICATIONS": "CERTIFICATIONS",
    "CERTIFICATION": "CERTIFICATIONS",
    "ACHIEVEMENTS": "ACHIEVEMENTS",
    "INTERNSHIP": "INTERNSHIP",
    "INTERNSHIPS": "INTERNSHIP",
    "INTERNSHIP EXPERIENCE": "INTERNSHIP",
}


def normalize_heading(line):
    compact = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
    compact = re.sub(r"\s+", " ", compact).strip()
    return HEADING_ALIASES.get(compact.upper())


def is_candidate_name_line(line):
    if not line:
        return False

    if "@" in line or re.search(r"\d", line):
        return False

    lower_line = line.lower()
    invalid_keywords = [
        "linkedin", "github", "email", "phone", "contact", "resume",
        "objective", "summary", "profile", "experience", "education",
        "skills", "project", "internship", "degree", "college",
        "university", "institute", "technology", "engineering",
        "developer", "analyst", "professional", "business", "solutions",
        "aspiring", "seeking", "motivated", "qualification", "objective"
    ]
    if any(keyword in lower_line for keyword in invalid_keywords):
        return False

    if normalize_heading(line):
        return False

    words = line.split()
    if len(words) < 2 or len(words) > 5:
        return False

    if any(len(word) < 2 for word in words):
        return False

    name_token = re.compile(r"^[A-Za-z.'-]+$")
    if not all(name_token.match(word) for word in words):
        return False

    if not all(word[0].isupper() for word in words if word[0].isalpha()):
        return False

    return True


def extract_candidate_name_from_filename(file_name):
    base = os.path.splitext(file_name)[0]
    segments = re.split(r"[-–—]+", base)
    candidate_part = segments[-1] if len(segments) > 1 else base
    candidate_part = re.sub(r"[_|]+", " ", candidate_part)
    candidate_part = re.sub(r"\b(resume|cv|profile|rag|main|project|updated|final|draft)\b", "", candidate_part, flags=re.IGNORECASE)
    candidate_part = re.sub(r"\s+", " ", candidate_part).strip()

    words = [w for w in candidate_part.split() if len(w) > 1]
    if len(words) >= 2:
        return " ".join(words).upper()
    return None


def extract_candidate_name(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:12]:
        if is_candidate_name_line(line):
            return line.upper()

    return "UNKNOWN"


def format_chunk(section, text):
    body = text.strip()

    if not body:
        return ""

    if section == "GENERAL":
        return body

    return f"{section}\n\n{body}"


def split_resume_sections(text):
    """
    Split resume into logical sections and include the heading inside each chunk.
    """

    chunks = []
    current_section = "GENERAL"
    current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        heading = normalize_heading(line)

        if heading:
            if current_lines:
                chunk_text = format_chunk(current_section, "\n".join(current_lines).strip())
                if chunk_text:
                    chunks.append({
                        "section": current_section,
                        "text": chunk_text
                    })

            current_section = heading
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_section != "GENERAL":
        chunk_text = format_chunk(current_section, "\n".join(current_lines).strip())
        if chunk_text:
            chunks.append({
                "section": current_section,
                "text": chunk_text
            })

    return chunks


def create_vector_store():

    embeddings = []
    metadata = []

    index_path = os.path.join(VECTOR_STORE_FOLDER, "resume_index.faiss")
    metadata_path = os.path.join(VECTOR_STORE_FOLDER, "metadata.pkl")

    if os.path.exists(index_path):
        os.remove(index_path)

    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    files = sorted([f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt")])

    if len(files) == 0:
        print("No extracted resume found.")
        return

    print(f"Found {len(files)} resume(s) to index.")

    for file_name in files:
        path = os.path.join(TEXT_FOLDER, file_name)

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        candidate_name = extract_candidate_name(text)

        if candidate_name == "UNKNOWN":
            fallback_candidate = extract_candidate_name_from_filename(file_name)
            if fallback_candidate:
                candidate_name = fallback_candidate
                print(f"Info: candidate_name extracted from filename fallback for {file_name} -> {candidate_name}")
            else:
                print(f"Warning: candidate_name extraction failed for {file_name}. Using UNKNOWN candidate_name.")

        chunks = split_resume_sections(text)

        print(f"Processing: {file_name} -> {len(chunks)} chunk(s) [candidate={candidate_name}]")

        for chunk in chunks:
            chunk_text = chunk["text"].strip()

            if not chunk_text:
                continue

            embedding = model.encode(
                "Represent this passage for retrieval: " + chunk_text
            )

            embeddings.append(embedding)

            metadata.append({
                "file_name": file_name,
                "candidate_name": candidate_name,
                "section": chunk["section"],
                "text": chunk_text,
                "name": os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ").strip()
            })

            print(
                f"  chunk source_file={file_name} | candidate_name={candidate_name} | "
                f"section={chunk['section']} | preview={chunk_text[:200].replace('\n', ' ')}"
            )

    if not embeddings:
        print("No chunks were generated for indexing.")
        return

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as file:
        pickle.dump(metadata, file)

    print("\nVector Store Created Successfully!")
    print(f"Stored {len(metadata)} sections.")


if __name__ == "__main__":
    create_vector_store()
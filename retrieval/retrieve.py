import os
import pickle
import re
import faiss
from llm.generate_answer import generate_answer
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_STORE_FOLDER = "data/vector_store"
TEXT_FOLDER = "data/extracted_text"
TOP_K = 5
SEARCH_LIMIT = 50

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def normalize_text(text):
    normalized = re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()
    return re.sub(r"\s+", " ", normalized)


def name_matches_query(query, candidate_name):
    query_tokens = set(normalize_text(query).split())
    candidate_tokens = set(normalize_text(candidate_name).split())
    if not candidate_tokens or len(candidate_tokens) < 2:
        return False
    return candidate_tokens.issubset(query_tokens)


def extract_candidate_name_from_filename(file_name):
    base = os.path.splitext(file_name)[0]
    base = re.sub(r"[_|]+", " ", base)
    fragments = [part.strip() for part in re.split(r"[-–—]+", base) if part.strip()]
    if len(fragments) > 1:
        maybe_name = fragments[-1]
    else:
        maybe_name = base

    maybe_name = re.sub(r"\bresume\b|\bcv\b|\bprofile\b", "", maybe_name, flags=re.IGNORECASE).strip()
    words = [w for w in maybe_name.split() if len(w) > 1]
    if len(words) >= 2:
        return " ".join(words).upper()
    return None


def extract_possible_name_from_query(query):
    stopwords = {
        "what", "are", "is", "the", "a", "an", "of", "for", "to", "in",
        "on", "with", "and", "or", "about", "details", "resume", "candidate",
        "skills", "technical", "education", "experience", "internship", "project",
        "projects", "certifications", "certification", "summary", "details",
        "please", "show", "give", "tell", "me", "my", "your"
    }
    words = [w for w in re.findall(r"[A-Za-z']+", query.lower()) if w not in stopwords]
    if len(words) >= 2:
        return " ".join(words[:2]).upper()
    return None


def get_all_candidate_names(metadata):
    names = set()
    for entry in metadata:
        candidate = entry.get("candidate_name")
        if candidate and candidate != "UNKNOWN":
            names.add(candidate.strip().upper())
        elif entry.get("name"):
            names.add(entry["name"].strip().upper())

    if os.path.isdir(TEXT_FOLDER):
        for file_name in os.listdir(TEXT_FOLDER):
            if not file_name.endswith(".txt"):
                continue
            parsed_name = extract_candidate_name_from_filename(file_name)
            if parsed_name:
                names.add(parsed_name)

    return names


def find_candidate_name(query, metadata):
    metadata_names = sorted(
        {entry.get("candidate_name", "UNKNOWN").strip().upper()
         for entry in metadata if entry.get("candidate_name") and entry.get("candidate_name") != "UNKNOWN"},
        key=lambda name: len(name.split()),
        reverse=True,
    )

    for candidate in metadata_names:
        if name_matches_query(query, candidate):
            return candidate

    all_names = sorted(
        get_all_candidate_names(metadata),
        key=lambda name: len(name.split()),
        reverse=True,
    )

    for candidate in all_names:
        if name_matches_query(query, candidate):
            return candidate

    possible_name = extract_possible_name_from_query(query)
    if possible_name:
        for candidate in metadata_names + [name for name in all_names if name not in metadata_names]:
            if name_matches_query(possible_name, candidate):
                return candidate

    return None


def get_section_priority(query):
    lowered = query.lower()

    if "skill" in lowered or "skills" in lowered:
        return ["TECHNICAL SKILLS", "SKILLS"]
    if "project" in lowered:
        return ["PROJECTS"]
    if "education" in lowered or "degree" in lowered or "qualification" in lowered:
        return ["EDUCATION"]
    if "experience" in lowered or "internship" in lowered:
        return ["EXPERIENCE", "INTERNSHIP"]

    return None


def filter_and_sort_results(results, query, candidate_name=None):
    priority_sections = get_section_priority(query)

    if candidate_name and priority_sections:
        section_results = [item for item in results if item.get("section") in priority_sections]
        if section_results:
            section_order = {section: i for i, section in enumerate(priority_sections)}
            return sorted(
                section_results,
                key=lambda item: (
                    section_order.get(item.get("section"), len(priority_sections)),
                    item["distance"],
                ),
            )
        if "skill" in query.lower() or "skills" in query.lower():
            return [{"error": "I found the candidate, but the technical skills section is not available."}]

    if priority_sections:
        return sorted(
            results,
            key=lambda item: (
                0 if item.get("section") in priority_sections else 1,
                item["distance"],
            ),
        )

    return sorted(results, key=lambda item: item["distance"])


def retrieve_resume(query, top_k=TOP_K):
    index_path = os.path.join(VECTOR_STORE_FOLDER, "resume_index.faiss")
    metadata_path = os.path.join(VECTOR_STORE_FOLDER, "metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("Vector store not found. Rebuild it first.")
        return []

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as file:
        metadata = pickle.load(file)

    candidate_name = find_candidate_name(query, metadata)
    if candidate_name:
        print(f"Candidate filter applied: {candidate_name}")

    search_limit = min(SEARCH_LIMIT, len(metadata))

    query_embedding = model.encode(
        "Represent this sentence for searching relevant passages: " + query
    )
    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, search_limit)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        entry = dict(metadata[idx])
        entry["distance"] = float(distance)
        results.append(entry)

    if candidate_name:
        results = [item for item in results if item.get("candidate_name") == candidate_name]
        if not results:
            return [{"error": "I could not find this candidate in the uploaded resumes."}]

    else:
        possible_name = extract_possible_name_from_query(query)
        if possible_name:
            all_candidates = get_all_candidate_names(metadata)
            if any(name_matches_query(possible_name, candidate) for candidate in all_candidates):
                return [{"error": "I could not find this candidate in the uploaded resumes."}]

    results = filter_and_sort_results(results, query, candidate_name)

    if results and isinstance(results[0], dict) and results[0].get("error"):
        return results

    debug_results = results[:top_k]
    print("Retrieved candidate chunks:")
    for rank, item in enumerate(debug_results, start=1):
        print(
            f"{rank}. candidate_name={item.get('candidate_name')} | section={item.get('section')} | "
            f"source={item.get('file_name')} | content={item.get('text', '')[:300].replace('\n', ' ')}"
        )

    return debug_results[:top_k]


if __name__ == "__main__":

    query = input("Enter your question: ")

    results = retrieve_resume(query, top_k=3)

    if not results:
        print("No results found.")
    else:
        result = results[0]
        print("\nRetrieved Section")
        print("----------------------------")
        print(result["section"])
        print(result["text"])
        answer = generate_answer(
            query,
            result["text"]
        )

        print("\nFinal Answer")
        print("-" * 50)
        print(answer)
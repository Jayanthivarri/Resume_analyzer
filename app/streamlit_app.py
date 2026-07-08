import re
import sys
import os
import difflib
import streamlit as st
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from retrieval.retrieve import retrieve_resume
from llm.generate_answer import generate_answer
from evaluation.answer_metrics import semantic_similarity


def normalize_ground_truth_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_ground_truth_dataframe():
    excel_path = "data/groundtruth.csv/Rag_Responses.xlsx"

    if not os.path.exists(excel_path):
        return None, excel_path

    try:
        df = pd.read_excel(excel_path)
    except Exception:
        try:
            import openpyxl  # noqa: F401
            df = pd.read_excel(excel_path)
        except Exception:
            return None, excel_path

    if df is None or df.empty:
        return pd.DataFrame(columns=["Sample_query", "GroundTruth_Answer"]), excel_path

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    sample_col = None
    answer_col = None

    if "Sample_query" in df.columns and "GroundTruth_Answer" in df.columns:
        return df[["Sample_query", "GroundTruth_Answer"]].copy(), excel_path

    for column in df.columns:
        lowered = column.lower()
        if sample_col is None and ("sample" in lowered or "query" in lowered):
            sample_col = column
        if answer_col is None and ("ground" in lowered or "answer" in lowered):
            answer_col = column

    if sample_col and answer_col:
        df = df[[sample_col, answer_col]].copy()
        df.columns = ["Sample_query", "GroundTruth_Answer"]

    return df, excel_path


def find_ground_truth_match(query, df):
    if df is None or df.empty:
        return None, None

    if "Sample_query" not in df.columns or "GroundTruth_Answer" not in df.columns:
        return None, None

    normalized_query = normalize_ground_truth_text(query)
    normalized_samples = [normalize_ground_truth_text(item) for item in df["Sample_query"].fillna("")]

    exact_idx = None
    for idx, sample in enumerate(normalized_samples):
        if sample == normalized_query:
            exact_idx = idx
            break

    if exact_idx is not None:
        return df.iloc[exact_idx]["GroundTruth_Answer"], df.iloc[exact_idx]["Sample_query"]

    if not normalized_query:
        return None, None

    close_matches = difflib.get_close_matches(
        normalized_query,
        normalized_samples,
        n=1,
        cutoff=0.6,
    )

    if close_matches:
        match_index = normalized_samples.index(close_matches[0])
        return df.iloc[match_index]["GroundTruth_Answer"], df.iloc[match_index]["Sample_query"]

    return None, None


st.set_page_config(
    page_title="Resume Analyzer using RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Analyzer using RAG")

st.write("Ask any question about the resume collection.")

query = st.text_input(
    "Enter your question",
    placeholder="Example: What are Rohitha Mareddy's technical skills?"
)

show_eval = st.checkbox("Show Ground Truth & Similarity")

if st.button("Get Answer"):

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching Resume..."):
        results = retrieve_resume(query, top_k=2)

    if not results:
        st.error("No matching resume found.")
        st.stop()

    if isinstance(results[0], dict) and results[0].get("error"):

        st.error(
            results[0].get(
                "message",
                results[0].get("error")
            )
        )
        st.stop()

    best_result = results[0]

    st.subheader("📌 Best Retrieved Match")

    st.write(f"**Candidate:** {best_result.get('candidate_name','Unknown')}")
    st.write(f"**Section:** {best_result.get('section','-')}")
    st.write(f"**File:** {best_result.get('file_name','-')}")

    st.text_area(
        "Retrieved Context",
        best_result.get("text", ""),
        height=220
    )

    st.subheader("📚 Top Retrieved Chunks")

    context_chunks = []

    for i, item in enumerate(results, start=1):

        chunk_text = item.get("text", "")
        context_chunks.append(chunk_text)

        with st.expander(
            f"Chunk {i} • {item.get('section','GENERAL')}"
        ):

            st.write(f"**Candidate:** {item.get('candidate_name','Unknown')}")
            st.write(f"**Section:** {item.get('section','-')}")
            st.write(f"**File:** {item.get('file_name','-')}")
            st.write(chunk_text)

    context = "\n\n".join(context_chunks)

    with st.spinner("Generating Answer..."):

        answer = generate_answer(
            query,
            context
        )

    st.subheader("💡 Final Answer")

    st.success(answer)
# =====================================================
# Ground Truth Evaluation
# =====================================================

if show_eval:

    excel_path = "data/groundtruth.csv/Rag_Responses.xlsx"

    if os.path.exists(excel_path):

        df = pd.read_excel(excel_path)

        # Safe conversion
        df["Sample_query"] = (
            df["Sample_query"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        user_query = query.lower().strip()

        match = df[
            df["Sample_query"] == user_query
        ]

        if not match.empty:

            ground_truth = str(
                match.iloc[0]["GroundTruth_Answer"]
            )

            score = semantic_similarity(
                answer,
                ground_truth
            )

            st.subheader("✅ Ground Truth")

            st.info(ground_truth)

            st.subheader("📊 Evaluation")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Semantic Similarity",
                    f"{score:.4f}"
                )

            with col2:

                if score >= 0.90:
                    st.success("Excellent Match ✅")

                elif score >= 0.75:
                    st.info("Good Match 👍")

                elif score >= 0.60:
                    st.warning("Average Match ⚠️")

                else:
                    st.error("Poor Match ❌")

        else:

            st.warning(
                "Ground truth not found for this question."
            )

    else:

        st.error(
            "Ground truth Excel file not found."
        )
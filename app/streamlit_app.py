import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrieval.retrieve import retrieve_resume
from llm.generate_answer import generate_answer

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

if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching Resume..."):
            results = retrieve_resume(query, top_k=3)

        if not results:
            st.error("No matching context was found. Rebuild the vector store first.")
            st.stop()

        if isinstance(results[0], dict) and results[0].get("error"):
            st.error(results[0]["error"])
            st.stop()

        best_result = results[0]

        st.subheader("Retrieved Section")
        st.write("**Candidate:**", best_result.get("candidate_name", "Unknown"))
        st.write("**Best Match Section:**", best_result["section"])
        st.write("**File:**", best_result.get("file_name", "-"))

        st.text_area(
            "Context",
            best_result["text"],
            height=220
        )

        st.caption("Top matching sections")
        for index, result in enumerate(results, start=1):
            st.write(f"{index}. **{result['section']}** — {result.get('candidate_name', '-')}: {result.get('file_name', '-')}")

        with st.spinner("Generating Answer..."):
            answer = generate_answer(
                query,
                best_result["text"]
            )

        st.subheader("Answer")
        st.success(answer)
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

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching Resume..."):
        results = retrieve_resume(query, top_k=2)

    if not results:
        st.error("No matching resume found.")
        st.stop()

    # Handle errors
    if isinstance(results[0], dict) and results[0].get("error"):

        message = results[0].get(
            "message",
            results[0].get("error", "Unknown error.")
        )

        st.error(message)
        st.stop()

    best_result = results[0]

    st.subheader("Retrieved Best Match")

    st.write(f"**Candidate:** {best_result.get('candidate_name', 'Unknown')}")
    st.write(f"**Section:** {best_result.get('section', '-')}")
    st.write(f"**File:** {best_result.get('file_name', '-')}")

    st.text_area(
        "Best Retrieved Context",
        best_result.get("text", ""),
        height=220
    )

    st.subheader("Top Retrieved Chunks")

    for i, item in enumerate(results, start=1):

        with st.expander(f"Chunk {i} - {item.get('section', '-')}"):

            st.write(f"**Candidate:** {item.get('candidate_name', '-')}")
            st.write(f"**File:** {item.get('file_name', '-')}")
            st.write(item.get("text", ""))

    # Combine Top-3 chunks
    context = "\n\n".join(
        item["text"] for item in results
    )

    with st.spinner("Generating Answer..."):

        answer = generate_answer(
            query,
            context
        )

    st.subheader("Answer")

    st.success(answer)
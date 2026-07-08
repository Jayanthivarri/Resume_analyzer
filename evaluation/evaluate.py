import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd

from retrieval.retrieve import retrieve_resume
from llm.generate_answer import generate_answer
from evaluation.answer_metrics import semantic_similarity

GROUND_TRUTH_FILE = "data/groundtruth.csv/Rag_Responses.xlsx"


def evaluate():

    df = pd.read_excel(GROUND_TRUTH_FILE)

    total_score = 0
    valid_count = 0

    for index, row in df.iterrows():

        question = row["Sample_query"]
        ground_truth = row["GroundTruth_Answer"]

        # Skip empty question
        if pd.isna(question):
            continue

        question = str(question).strip()

        if question == "":
            continue

        # Skip empty ground truth
        if pd.isna(ground_truth):
            continue

        ground_truth = str(ground_truth).strip()

        if ground_truth == "":
            continue

        # Retrieve
        results = retrieve_resume(question, top_k=2)

        if not results or results[0].get("error"):

            predicted = "No answer found"

        else:

            context = "\n\n".join(
                item["text"] for item in results
            )

            predicted = generate_answer(
                question,
                context
            )

        # Similarity
        score = semantic_similarity(
            predicted,
            ground_truth
        )

        total_score += score
        valid_count += 1

        print("=" * 80)
        print(f"Question      : {question}")
        print(f"Ground Truth  : {ground_truth}")
        print(f"Prediction    : {predicted}")
        print(f"Similarity    : {score:.4f}")

    print("\n" + "=" * 80)

    if valid_count == 0:
        print("No valid evaluation samples found.")
    else:
        average = total_score / valid_count
        print(f"Total Samples              : {valid_count}")
        print(f"Average Semantic Similarity: {average:.4f}")


if __name__ == "__main__":
    evaluate()
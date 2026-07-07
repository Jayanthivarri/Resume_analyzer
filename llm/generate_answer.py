import os
import google.generativeai as genai
from dotenv import load_dotenv
from llm.prompt import SYSTEM_PROMPT

# Load API Key
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_answer(question, context):

    prompt = f"""
{SYSTEM_PROMPT}

Resume Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)

    return response.text


if __name__ == "__main__":

    sample_context = """
Education

Bachelor of Technology in Electrical and Electronics Engineering

CGPA : 8.32
"""

    sample_question = "What is the candidate's qualification?"

    answer = generate_answer(sample_question, sample_context)

    print(answer)
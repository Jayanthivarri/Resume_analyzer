import os
import google.generativeai as genai
from dotenv import load_dotenv
from llm.prompt import SYSTEM_PROMPT

# Load API Key
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash-lite")


def generate_answer(question, context):

    prompt = f"""
{SYSTEM_PROMPT}

The following are the retrieved sections from a candidate's resume.

Resume Context:
{context}

Question:
{question}

Instructions:
- Answer ONLY using the provided resume context.
- Combine information from all retrieved sections if needed.
- If the answer is not available in the context, reply:
  "The requested information is not available in the resume."
- Do not guess or add external information.
- Keep the answer clear and concise.

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
import os
import fitz  # PyMuPDF

# Folder paths
RESUME_FOLDER = "data/resumes"
OUTPUT_FOLDER = "data/extracted_text"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file while preserving readable section breaks.
    """
    document = fitz.open(pdf_path)
    text = ""

    for page in document:
        text += page.get_text() + "\n"

    document.close()
    return text.replace("\r\n", "\n").strip()


def process_resumes():
    """
    Process all resumes in the resumes folder.
    """
    for file_name in os.listdir(RESUME_FOLDER):

        if file_name.endswith(".pdf"):

            pdf_path = os.path.join(RESUME_FOLDER, file_name)

            text = extract_text_from_pdf(pdf_path)

            output_file = os.path.splitext(file_name)[0] + ".txt"
            output_path = os.path.join(OUTPUT_FOLDER, output_file)

            with open(output_path, "w", encoding="utf-8") as file:
                file.write(text)

            print(f"Extracted: {file_name}")


if __name__ == "__main__":
    process_resumes()
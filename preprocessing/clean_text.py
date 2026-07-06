import os
import re

# Folder paths
INPUT_FOLDER = "data/extracted_text"
OUTPUT_FOLDER = "data/extracted_text"


def clean_text(text):
    """
    Clean extracted resume text.
    """

    # Remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def clean_all_files():
    """
    Clean all extracted text files.
    """

    for file_name in os.listdir(INPUT_FOLDER):

        if file_name.endswith(".txt"):

            file_path = os.path.join(INPUT_FOLDER, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            cleaned_text = clean_text(text)

            with open(file_path, "w", encoding="utf-8") as file:
                file.write(cleaned_text)

            print(f"Cleaned: {file_name}")


if __name__ == "__main__":
    clean_all_files()
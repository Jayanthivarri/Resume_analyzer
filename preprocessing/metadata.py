import os

RESUME_FOLDER = "data/resumes"

def create_metadata():
    metadata_list = []

    for file_name in os.listdir(RESUME_FOLDER):

        if file_name.endswith(".pdf"):

            metadata = {
                "name": os.path.splitext(file_name)[0],
                "file_name": file_name,
                "document_type": "resume"
            }

            metadata_list.append(metadata)

    return metadata_list


if __name__ == "__main__":
    metadata = create_metadata()

    print("Metadata Created Successfully\n")

    for item in metadata:
        print(item)
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.vector_store import split_resume_sections
from retrieval.retrieve import find_candidate_name


class SectionChunkingTests(unittest.TestCase):
    def test_split_resume_sections_includes_section_heading_in_chunk_text(self):
        resume_text = """Name
Contact
EDUCATION
Bachelor of Technology
TECHNICAL SKILLS
Python, SQL, ML
PROJECTS
Built a resume analyzer"""

        chunks = split_resume_sections(resume_text)

        technical_chunk = next(
            chunk for chunk in chunks if chunk["section"] == "TECHNICAL SKILLS"
        )

        self.assertTrue(technical_chunk["text"].startswith("TECHNICAL SKILLS"))
        self.assertIn("Python", technical_chunk["text"])

    def test_find_candidate_name_matches_out_of_order_name_tokens(self):
        query = "What are Rohitha Mareddy's technical skills?"
        metadata = [
            {"candidate_name": "MAREDDY ROHITHA"},
            {"candidate_name": "AVINASH CHAVA"},
        ]

        matched = find_candidate_name(query, metadata)

        self.assertEqual(matched, "MAREDDY ROHITHA")


if __name__ == "__main__":
    unittest.main()

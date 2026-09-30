import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath("backend"))

from app.services.segmenter import QuestionSegmenter

def test_question_segmentation():
    sample_ocr_lines = [
        "Registration No: 917720104001",
        "B.E. DEGREE EXAMINATIONS, APRIL/MAY 2023",
        "Fourth Semester - Computer Science and Engineering",
        "CS8492 - DATABASE MANAGEMENT SYSTEMS",
        "Time : Three Hours                                Maximum : 100 Marks",
        "",
        "PART A - (10 x 2 = 20 Marks)",
        "1. (a) Define B-Tree and state its balance properties. [7 Marks]",
        "(b) Differentiate between dense and sparse index. (3M)",
        "OR",
        "2. Explain the ACID properties of a transaction with suitable banking examples. [10 Marks]",
        "",
        "PART B - (5 x 13 = 65 Marks)",
        "UNIT II",
        "3. (a) What is 3NF? How does it differ from BCNF? Explain with a relation R(A,B,C,D). [13 Marks]",
        "OR",
        "4. Describe the Two-Phase Locking (2PL) protocol and explain how it prevents concurrency anomalies. [13 Marks]"
    ]

    print("=" * 60)
    print("  Testing QuestionSegmenter on Real-World Exam Format")
    print("=" * 60)

    questions = QuestionSegmenter.parse_page_text(sample_ocr_lines, page_number=1)
    
    print(f"\nExtracted {len(questions)} structured question records:\n")
    for idx, q in enumerate(questions, start=1):
        print(f"[{idx}] Question Number : {q['question_number']}")
        print(f"    Syllabus Unit   : {q['unit_module']}")
        print(f"    Marks           : {q['marks']}")
        print(f"    Clean Text      : {q['question_text']}")
        print("-" * 50)

if __name__ == "__main__":
    test_question_segmentation()

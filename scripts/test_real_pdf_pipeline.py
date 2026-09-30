import sys
import os
from pathlib import Path

# Add backend to sys.path
sys.path.insert(0, os.path.abspath("backend"))

from app.services.ocr_engine import OCREngine
from app.services.segmenter import QuestionSegmenter

def test_sample_pdf(pdf_filename: str):
    pdf_path = Path("data/samples") / pdf_filename
    print("=" * 70)
    print(f"  Testing Pipeline on Real PDF: {pdf_filename}")
    print("=" * 70)
    
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        return

    try:
        engine = OCREngine()
        print(f"\n[1/3] Loading and preprocessing document: {pdf_filename}...")
        pages_data = engine.process_document(pdf_path)
        print(f"      Successfully processed {len(pages_data)} page(s).")
        
        for p in pages_data:
            print(f"      - Page {p['page_number']}: {p['word_count']} words, Avg Confidence: {p['confidence']}%")
            if p['diagrams']:
                print(f"        Detected {len(p['diagrams'])} diagram(s): {[d['path'] for d in p['diagrams']]}")

        print(f"\n[2/3] Segmenting extracted text into questions...")
        questions = QuestionSegmenter.segment_document(pages_data)
        print(f"      Successfully segmented {len(questions)} question(s).\n")

        print("[3/3] Segmented Questions Output:")
        print("=" * 70)
        for idx, q in enumerate(questions, start=1):
            print(f"[{idx}] Question Number : {q['question_number']}")
            print(f"    Syllabus Unit   : {q.get('unit_module', 'N/A')}")
            print(f"    Marks           : {q.get('marks', 'N/A')}")
            if q.get('image_ref'):
                print(f"    Diagram Image   : {q['image_ref']}")
            print(f"    Text            : {q['question_text']}")
            print("-" * 70)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed on {pdf_filename}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    sample_file = "3rdBE_DesignOfOperatingSystem.pdf"
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]
    test_sample_pdf(sample_file)

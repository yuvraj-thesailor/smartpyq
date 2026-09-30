import sys
import os
import shutil

print("=" * 60)
print("  SmartPYQ Environment & Pipeline Quickstart Verification")
print("=" * 60)
print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")

# 1. Test Core Library Imports
print("\n[1/5] Testing Library Imports...")
libraries = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("sqlalchemy", "SQLAlchemy"),
    ("pgvector", "pgvector-python"),
    ("pydantic", "Pydantic"),
    ("cv2", "OpenCV"),
    ("PIL", "Pillow"),
    ("pytesseract", "PyTesseract"),
    ("sentence_transformers", "Sentence Transformers"),
    ("yake", "YAKE Keyword Extractor"),
]

missing_libs = []
for module_name, display_name in libraries:
    try:
        __import__(module_name)
        print(f"  [OK] {display_name}")
    except ImportError as e:
        print(f"  [MISSING] {display_name} ({e})")
        missing_libs.append(module_name)

if missing_libs:
    print(f"\n[!] Please install missing libraries using:")
    print(f"    pip install -r backend/requirements.txt")

# 2. Test Tesseract OCR Executable
print("\n[2/5] Testing Tesseract OCR...")
try:
    import pytesseract
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        version = pytesseract.get_tesseract_version()
        print(f"  [OK] Tesseract found at: {tesseract_path} (Version: {version})")
    else:
        # Check standard Windows paths
        win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(win_default):
            pytesseract.pytesseract.tesseract_cmd = win_default
            version = pytesseract.get_tesseract_version()
            print(f"  [OK] Tesseract found at: {win_default} (Version: {version})")
        else:
            print("  [WARN] Tesseract executable not found in system PATH.")
            print("         If on Windows, install UB-Mannheim Tesseract OCR installer")
            print("         and add it to PATH or set TESSERACT_CMD in backend/.env")
except Exception as e:
    print(f"  [WARN] Tesseract check error: {e}")

# 3. Test Sentence Transformers & Embedding Generation
print("\n[3/5] Testing Sentence-Transformers (all-MiniLM-L6-v2)...")
try:
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"  Loading model '{model_name}'...")
    model = SentenceTransformer(model_name)
    sample_text = "Explain Dijkstra's shortest path algorithm with time complexity."
    embedding = model.encode(sample_text)
    print(f"  [OK] Generated embedding successfully.")
    print(f"       Dimension: {len(embedding)} (Expected: 384)")
    print(f"       Sample slice: {embedding[:5].round(4)}")
except Exception as e:
    print(f"  [ERROR] Failed to load/run SentenceTransformer: {e}")

# 4. Test YAKE Keyword Extraction
print("\n[4/5] Testing Keyword Extraction (YAKE)...")
try:
    import yake
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=3)
    sample_q = "Discuss the ACID properties in Database Management Systems with suitable examples."
    keywords = [kw[0] for kw in kw_extractor.extract_keywords(sample_q)]
    print(f"  Sample Question: '{sample_q}'")
    print(f"  [OK] Auto-extracted tags: {keywords}")
except Exception as e:
    print(f"  [ERROR] YAKE test error: {e}")

# 5. Test Synthetic OCR Image Extraction
print("\n[5/5] Testing Synthetic Image OCR Generation...")
try:
    from PIL import Image, ImageDraw, ImageFont
    import pytesseract
    
    # Create a blank white image with black text
    img = Image.new('RGB', (600, 150), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    test_text = "Q1. (a) State the Master Theorem for divide and conquer recurrence relations."
    d.text((20, 50), test_text, fill=(0, 0, 0))
    
    # Run OCR on the synthetic image
    ocr_result = pytesseract.image_to_string(img).strip()
    print(f"  Generated Text: '{test_text}'")
    print(f"  OCR Output:     '{ocr_result}'")
    if "Master Theorem" in ocr_result:
        print("  [OK] OCR synthetic test PASSED.")
    else:
        print("  [WARN] OCR output did not exactly match expected text, check Tesseract settings.")
except Exception as e:
    print(f"  [NOTE] Synthetic OCR test skipped or failed: {e}")

print("\n" + "=" * 60)
print("  Quickstart Verification Completed!")
print("=" * 60)

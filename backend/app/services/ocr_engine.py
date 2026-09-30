import os
import uuid
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
import pytesseract
from pytesseract import Output

from app.core.config import settings
from app.services.preprocessor import ImagePreprocessor

# Configure Tesseract executable path if specified in settings or default Windows location
if settings.TESSERACT_CMD and os.path.exists(settings.TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
elif os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCREngine:
    """
    Layout-aware OCR and Document Ingestion Engine.
    Handles:
    - Multi-page PDF to image conversion
    - Tesseract OCR with Page Segmentation Mode (PSM) tuning
    - Line and word-level bounding-box and confidence scoring
    - Diagram / circuit region cropping and storage
    """

    def __init__(self, diagram_dir: Optional[Union[str, Path]] = None):
        self.diagram_dir = Path(diagram_dir or os.path.join(settings.UPLOAD_DIR, "diagrams"))
        self.diagram_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_document_pages(file_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Loads an uploaded PDF or image file and returns a list of page images in OpenCV BGR format.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            return [img]

        elif ext == ".pdf":
            try:
                # Primary: PyMuPDF (fitz) - high performance, 300 DPI, zero external binaries needed
                import fitz
                doc = fitz.open(str(path))
                images = []
                # Zoom matrix for ~300 DPI rendering (72 DPI * 4.167 ≈ 300 DPI)
                zoom = 300 / 72
                mat = fitz.Matrix(zoom, zoom)
                for page in doc:
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, 3))
                    # Convert RGB to OpenCV BGR
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    images.append(img_bgr)
                return images
            except Exception as e_fitz:
                try:
                    from pdf2image import convert_from_path
                    pil_pages = convert_from_path(str(path), dpi=300)
                    return [ImagePreprocessor.to_cv2_image(p) for p in pil_pages]
                except Exception:
                    raise RuntimeError(
                        f"PDF page rendering failed: {e_fitz}. Ensure PyMuPDF is installed."
                    )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def extract_text_and_confidence(self, image: np.ndarray, psm: int = 6) -> Dict[str, Any]:
        """
        Runs Tesseract OCR using Page Segmentation Mode (PSM).
        PSM 6: Assume a single uniform block of text (ideal for structured exam questions).
        PSM 3: Fully automatic page segmentation (fallback for complex multi-column).
        """
        # Run Tesseract with data dictionary output
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config=f"--psm {psm} --oem 3"
        )

        n_boxes = len(data["text"])
        lines_dict: Dict[Tuple[int, int], List[str]] = {}
        confidences: List[float] = []

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])

            if text:  # Non-empty token
                block_num = data["block_num"][i]
                line_num = data["line_num"][i]
                key = (block_num, line_num)

                if key not in lines_dict:
                    lines_dict[key] = []
                lines_dict[key].append(text)

                if conf > 0:  # Valid confidence score
                    confidences.append(conf)

        # Assemble joined text lines in reading order
        assembled_lines = [" ".join(words) for words in lines_dict.values()]
        full_text = "\n".join(assembled_lines)
        avg_confidence = round(float(np.mean(confidences)), 2) if confidences else 0.0

        return {
            "text": full_text,
            "lines": assembled_lines,
            "confidence": avg_confidence,
            "word_count": len(confidences)
        }

    def crop_and_save_diagram(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Crops a non-text visual block (circuit, diagram, chemical structure)
        and saves it to disk with a unique UUID filename.
        """
        x, y, w, h = bbox
        # Add slight padding if within bounds
        ih, iw = image.shape[:2]
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(iw, x + w + pad), min(ih, y + h + pad)

        cropped = image[y1:y2, x1:x2]
        filename = f"diagram_{uuid.uuid4().hex[:12]}.png"
        filepath = self.diagram_dir / filename

        cv2.imwrite(str(filepath), cropped)
        return str(filepath.relative_to(Path(settings.UPLOAD_DIR).parent) if Path(settings.UPLOAD_DIR).parent in filepath.parents else filepath)

    def process_page(self, page_image: np.ndarray, page_number: int = 1) -> Dict[str, Any]:
        """
        Orchestrates full page processing:
        1. OpenCV preprocessing (deskew, shadow removal, adaptive binarization, region detection)
        2. Tesseract OCR on cleaned binary
        3. Cropping and saving any diagram regions
        """
        # Step 1: Preprocess image
        prep_results = ImagePreprocessor.process_page(page_image)
        clean_binary = prep_results["binary_image"]
        deskewed_orig = prep_results["deskewed_image"]
        regions = prep_results["regions"]

        # Step 2: OCR extraction
        ocr_data = self.extract_text_and_confidence(clean_binary, psm=6)
        
        # If confidence is low or text is too sparse with PSM 6, try fallback PSM 3
        if ocr_data["word_count"] < 10 or ocr_data["confidence"] < 40.0:
            fallback_data = self.extract_text_and_confidence(clean_binary, psm=3)
            if fallback_data["word_count"] > ocr_data["word_count"]:
                ocr_data = fallback_data

        # Step 3: Crop and save diagram regions
        diagram_paths = []
        for reg in regions:
            if reg.get("is_diagram", False):
                diag_path = self.crop_and_save_diagram(deskewed_orig, reg["bbox"])
                diagram_paths.append({
                    "path": diag_path,
                    "bbox": reg["bbox"]
                })

        return {
            "page_number": page_number,
            "raw_text": ocr_data["text"],
            "lines": ocr_data["lines"],
            "confidence": ocr_data["confidence"],
            "word_count": ocr_data["word_count"],
            "skew_angle": prep_results["skew_angle"],
            "diagrams": diagram_paths
        }

    def process_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Processes all pages in a document (PDF or single image) and returns
        a list of structured page dictionaries.
        """
        pages = self.load_document_pages(file_path)
        results = []
        for idx, page_img in enumerate(pages, start=1):
            page_res = self.process_page(page_img, page_number=idx)
            results.append(page_res)
        return results

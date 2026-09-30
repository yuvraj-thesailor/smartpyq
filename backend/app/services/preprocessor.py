import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any, Union
from pathlib import Path


class ImagePreprocessor:
    """
    Computer Vision pipeline for normalizing scanned/photographed question papers.
    Features:
    - Shadow removal & contrast enhancement (CLAHE)
    - Adaptive binarization (Otsu & Gaussian Thresholding)
    - Automatic text deskewing (angle detection & rotation)
    - Layout & diagram region detection
    """

    @staticmethod
    def to_cv2_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Converts various image input formats (file path, PIL Image, numpy array) to OpenCV BGR format."""
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not load image from path: {image_input}")
            return img
        elif isinstance(image_input, Image.Image):
            # Convert PIL RGB to OpenCV BGR
            rgb_arr = np.array(image_input)
            if len(rgb_arr.shape) == 2:  # Grayscale PIL Image
                return cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detects text skew angle in degrees and rotates the image straight.
        Useful for phone photographs taken at slight angles.
        """
        # Convert to grayscale if not already
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Invert colors: text becomes white (255), background black (0)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Find coordinates of all foreground (text) pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            # Not enough text pixels to compute reliable angle
            return image, 0.0

        # Compute minimum-area bounding box of all text pixels
        angle = cv2.minAreaRect(coords)[-1]

        # Normalize OpenCV's angle to standard -45 to +45 degrees
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # If angle is negligible (< 0.5 deg), avoid unnecessary interpolation
        if abs(angle) < 0.5:
            return image, 0.0

        # Rotate image around its center
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated, angle

    @staticmethod
    def enhance_and_binarize(image: np.ndarray) -> np.ndarray:
        """
        Removes shadows and uneven illumination (common in phone camera shots),
        applies Contrast Limited Adaptive Histogram Equalization (CLAHE),
        and returns a clean, high-contrast binary image (black text on white).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Apply CLAHE to balance contrast across uneven shadows
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # Bilateral filter removes paper grain noise while preserving sharp font edges
        denoised = cv2.bilateralFilter(contrast_enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # Otsu's thresholding automatically calculates optimal threshold value
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return binary

    @staticmethod
    def detect_regions(binary_image: np.ndarray, min_area: int = 500) -> List[Dict[str, Any]]:
        """
        Analyzes page layout to detect text blocks and isolate non-text regions
        (diagrams, circuits, tables, chemical structures).
        Returns bounding boxes sorted in top-to-bottom reading order.
        """
        h, w = binary_image.shape[:2]

        # Invert: text/diagrams white, background black
        inv = cv2.bitwise_not(binary_image)

        # Morphological kernel to merge nearby characters into cohesive paragraph/diagram blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
        dilated = cv2.dilate(inv, kernel, iterations=2)

        # Find external contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, rw, rh = cv2.boundingRect(cnt)

            # Heuristic: diagrams/circuits usually have high aspect ratio + large surface area
            # compared to a single text line
            aspect_ratio = rw / float(rh) if rh > 0 else 0
            is_potential_diagram = (rw > w * 0.3 and rh > h * 0.15) or (0.5 < aspect_ratio < 2.0 and area > 15000)

            regions.append({
                "bbox": (x, y, rw, rh),
                "is_diagram": is_potential_diagram,
                "area": area
            })

        # Sort bounding boxes top-to-bottom, left-to-right (standard reading order)
        regions.sort(key=lambda r: (r["bbox"][1] // 30) * 30 + r["bbox"][0] // 100)
        return regions

    @classmethod
    def process_page(cls, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Executes the complete preprocessing pipeline on a single page:
        1. Formats input to OpenCV
        2. Auto-deskews
        3. Shadow-removal & binarization
        4. Layout & diagram region extraction
        """
        orig_img = cls.to_cv2_image(image_input)
        deskewed_img, skew_angle = cls.deskew(orig_img)
        binary_img = cls.enhance_and_binarize(deskewed_img)
        regions = cls.detect_regions(binary_img)

        return {
            "original_image": orig_img,
            "deskewed_image": deskewed_img,
            "binary_image": binary_img,
            "skew_angle": round(skew_angle, 2),
            "regions": regions
        }

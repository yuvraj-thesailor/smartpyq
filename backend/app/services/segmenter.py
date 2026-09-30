import re
from typing import List, Dict, Any, Optional, Tuple


class QuestionSegmenter:
    """
    Question Segmentation & Metadata Parsing Service.
    Splits continuous OCR text from exam papers into structured, individual question records.
    
    Features:
    - Multi-pattern question numbering detection (Q1, 1., 1(a), (i), (a), PART A, OR choices)
    - Marks weightage extraction ([7 Marks], (10M), 16 Marks)
    - Syllabus Unit / Module / Section detection (UNIT-I, MODULE 2, PART B)
    - Diagram association to nearest question
    - Header & footer noise removal (e.g. 'Page 1 of 4', 'Reg. No:', etc.)
    """

    # Common exam header/footer noise patterns to ignore
    NOISE_PATTERNS = [
        r'^\s*Reg(?:istration)?\.?\s*No\.?\s*[:\-_]?.*$',
        r'^\s*Roll\s*No\.?\s*[:\-_]?.*$',
        r'^\s*Page\s+\d+\s+of\s+\d+\s*$',
        r'^\s*Total\s+No\.?\s+of\s+Pages?\s*[:\-_]?\s*\d+.*$',
        r'^\s*Time\s*:\s*\d+.*Hours?.*$',
        r'^\s*Max(?:imum)?\.?\s*Marks?\s*:\s*\d+.*$',
        r'^\s*B\.?E\.?|B\.?Tech\.?|M\.?E\.?|M\.?Tech\.?|B\.?Sc\.?|B\.?C\.?A\.?.*Degree\s+Examinations?.*$',
        r'^\s*[A-Za-z0-9]+-[A-Za-z0-9]+\s+\d+\s*(?:\(Contd\.\))?\s*$', # e.g. "A-316 1 (Contd.)"
        r'^\s*\(Contd\.\)\s*$',
    ]

    # Regex patterns for question numbering
    QUESTION_START_PATTERNS = [
        # Match 'Q1.', 'Q.1', 'Question 1', 'Q 1(a)', 'Q. 1 (a)', '4, (a)', '4. (a)'
        re.compile(r'^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})\s*(?:[\.\,\)\:\-])\s*(?:\(([a-zA-Z]|[ivxIVX]+)\)|([a-zA-Z]))\s*[\.\,\:\-]?\s*(.*)$', re.IGNORECASE),
        # Match '1.', '1)', '1 -', '1,'
        re.compile(r'^\s*(\d{1,3})\s*(?:[\.\,\)\:\-])\s+(.*)$'),
        # Match sub-question alone '(a)', '(i)', 'a)', 'i)', '(b)'
        re.compile(r'^\s*(?:\(([a-zA-Z]|[ivxIVX]+)\)|([a-zA-Z]|[ivxIVX]+)\))\s*[\.\,\:\-]?\s*(.*)$'),
    ]

    # Section / Unit / Part patterns
    SECTION_PATTERN = re.compile(
        r'^\s*(?:PART|SECTION|UNIT|MODULE)\s*[-–:]?\s*([A-Za-z0-9IVX]+)\s*(?:[-–:].*)?$',
        re.IGNORECASE
    )

    # Patterns for extracting marks e.g. [7 Marks], (10M), 16 marks, or standalone trailing marks like "... 7"
    MARKS_PATTERNS = [
        re.compile(r'\[(?:[^\]]*?)?(\d{1,2})\s*(?:marks?|m)\b.*?\]', re.IGNORECASE),
        re.compile(r'\((?:[^\)]*?)?(\d{1,2})\s*(?:marks?|m)\b.*?\)', re.IGNORECASE),
        re.compile(r'\b(\d{1,2})\s*(?:marks?|mks)\b', re.IGNORECASE),
        re.compile(r'\b(\d{1,2})\s*M\b'),
        re.compile(r'\s+(\d{1,2})\s*$'),  # Standalone mark at end of question line e.g. "... 7"
    ]

    # 'OR' separator detection
    OR_PATTERN = re.compile(r'^\s*(?:---\s*)?\(?\s*OR\s*\)?(?:\s*---)?\s*$', re.IGNORECASE)

    @classmethod
    def is_noise_line(cls, line: str) -> bool:
        """Checks if a line contains header, footer, or administrative metadata."""
        for pattern in cls.NOISE_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                return True
        return False

    @classmethod
    def extract_marks(cls, text: str) -> Tuple[Optional[int], str]:
        """
        Extracts mark weightage from question text and cleans the text of the mark annotation.
        Returns: (marks_int, cleaned_text)
        """
        extracted_mark = None
        cleaned_text = text

        for pattern in cls.MARKS_PATTERNS:
            match = pattern.search(cleaned_text)
            if match:
                try:
                    mark_val = int(match.group(1))
                    if 1 <= mark_val <= 100:  # Reasonable exam marks range
                        extracted_mark = mark_val
                        # Remove mark notation from question text for cleaner search indexing
                        cleaned_text = pattern.sub('', cleaned_text).strip()
                        break
                except (ValueError, IndexError):
                    continue

        return extracted_mark, cleaned_text.strip()

    @classmethod
    def detect_unit_or_module(cls, line: str) -> Optional[int]:
        """Converts Unit/Module Roman numerals or digits (e.g. 'UNIT-II' or 'MODULE 3') to an integer."""
        match = cls.SECTION_PATTERN.match(line)
        if match:
            val = match.group(1).upper()
            roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "A": 1, "B": 2, "C": 3}
            if val in roman_map:
                return roman_map[val]
            if val.isdigit():
                return int(val)
        return None

    @classmethod
    def parse_page_text(
        cls, 
        lines: List[str], 
        page_number: int = 1,
        page_diagrams: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Parses raw text lines of a single exam page into segmented question dictionaries.
        """
        questions: List[Dict[str, Any]] = []
        current_unit: Optional[int] = None
        current_question: Optional[Dict[str, Any]] = None
        is_or_choice: bool = False

        diagram_queue = list(page_diagrams or [])

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if this line is an administrative noise header/footer
            if cls.is_noise_line(line_str):
                continue

            # Check for Unit / Section divider (e.g. "UNIT - II" or "PART B")
            detected_unit = cls.detect_unit_or_module(line_str)
            if detected_unit is not None:
                current_unit = detected_unit
                continue

            # Check for 'OR' delimiter (Question choice)
            if cls.OR_PATTERN.match(line_str):
                is_or_choice = True
                continue

            # Check if line marks the start of a new question or sub-question
            match_found = False
            q_label = None
            q_content = ""

            for pat in cls.QUESTION_START_PATTERNS:
                m = pat.match(line_str)
                if m:
                    groups = [g for g in m.groups() if g is not None]
                    if len(groups) >= 2:
                        # Case 1: Primary number + Sub letter e.g. "1" and "a" -> "1(a)"
                        if len(groups) == 3:
                            q_num, sub_letter, q_content = groups[0], groups[1], groups[2]
                            q_label = f"{q_num}({sub_letter})"
                        elif len(groups) == 2:
                            q_num, q_content = groups[0], groups[1]
                            q_label = str(q_num)
                        match_found = True
                        break
                    elif len(groups) == 1:
                        # Sub letter alone e.g. "(b)"
                        q_label = groups[0]
                        q_content = ""
                        match_found = True
                        break

            if match_found and q_label:
                # If there's an active question being accumulated, finalize it
                if current_question is not None:
                    # Finalize marks and text
                    marks, clean_txt = cls.extract_marks(current_question["raw_text"])
                    current_question["marks"] = marks
                    current_question["question_text"] = clean_txt
                    questions.append(current_question)

                # If this question followed an 'OR', record the flag
                final_q_num = f"{q_label} (OR)" if is_or_choice else q_label
                is_or_choice = False

                # Attach nearest diagram if available
                img_ref = None
                if diagram_queue:
                    diag = diagram_queue.pop(0)
                    img_ref = diag["path"]

                current_question = {
                    "question_number": final_q_num,
                    "raw_text": q_content if q_content else "",
                    "unit_module": current_unit,
                    "page_number": page_number,
                    "image_ref": img_ref,
                    "marks": None,
                    "question_text": ""
                }
            else:
                # Continuation of the current question's body
                if current_question is not None:
                    if current_question["raw_text"]:
                        current_question["raw_text"] += " " + line_str
                    else:
                        current_question["raw_text"] = line_str

        # Finalize the last remaining question on the page
        if current_question is not None and current_question["raw_text"].strip():
            marks, clean_txt = cls.extract_marks(current_question["raw_text"])
            current_question["marks"] = marks
            current_question["question_text"] = clean_txt
            questions.append(current_question)

        return questions

    @classmethod
    def segment_document(cls, document_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Segments all pages of a processed document and organizes them into
        a unified list of structured question records.
        """
        all_questions: List[Dict[str, Any]] = []

        for page in document_pages:
            lines = page.get("lines", [])
            page_num = page.get("page_number", 1)
            diagrams = page.get("diagrams", [])

            page_questions = cls.parse_page_text(
                lines=lines,
                page_number=page_num,
                page_diagrams=diagrams
            )
            all_questions.extend(page_questions)

        return all_questions

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Literal, Tuple

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import language_tool_python

Variant = Literal["us", "uk", "both"]
IssueType = Literal["spelling", "grammar", "punctuation"]
Severity = Literal["major", "minor"]

OCR_UNCLEAR_CONF_THRESHOLD = 60  # word conf below this => [unclear]
MAX_REPLACEMENTS = 5

# If Windows + tesseract not in PATH, uncomment:
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

app = FastAPI(title="OCR + Grammar/Spelling API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def variant_to_lt_language(variant: Variant) -> str:
    return "en-GB" if variant == "uk" else "en-US"


def safe_int(x: str, default: int = -1) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def is_probable_us_uk_variant(word: str) -> bool:
    # Minimal US/UK variant suppression when variant=both.
    pairs = {
        ("color", "colour"),
        ("organize", "organise"),
        ("organizing", "organising"),
        ("organization", "organisation"),
        ("center", "centre"),
        ("favorite", "favourite"),
        ("analyze", "analyse"),
        ("license", "licence"),
        ("traveler", "traveller"),
        ("canceled", "cancelled"),
    }
    w = word.lower()
    return any(w == a or w == b for a, b in pairs)


def line_number_from_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def apply_fixes(text: str, matches: List[Any]) -> str:
    """Apply first replacement of each match from end to start to keep offsets stable."""
    edits: List[Tuple[int, int, str]] = []
    for m in matches:
        repls = list(getattr(m, "replacements", []) or [])
        if not repls:
            continue
        edits.append((int(m.offset), int(m.errorLength), repls[0]))

    s = text
    for offset, length, repl in sorted(edits, key=lambda t: t[0], reverse=True):
        s = s[:offset] + repl + s[offset + length :]
    return s


def classify_issue(m: Any) -> Tuple[IssueType, Severity]:
    """Map LanguageTool match to our categories + severity."""
    rule_issue_type = (getattr(m, "ruleIssueType", "") or "").lower()
    cat_id = (getattr(getattr(m, "category", None), "id", "") or "").lower()
    msg = (getattr(m, "message", "") or "").lower()

    if rule_issue_type == "misspelling" or "typo" in cat_id or "misspell" in msg:
        return "spelling", "major"
    if "punct" in cat_id or "comma" in msg or "apostrophe" in msg or "quotation" in msg:
        return "punctuation", "minor"
    return "grammar", "major"


def compute_scores(
    issues: List[Dict[str, Any]],
    ocr_confidence: int,
    uncertain_count: int,
    total_words: int,
) -> Tuple[int, int, int, Dict[str, int]]:
    """Heuristic scoring (not a guarantee)."""
    major = sum(1 for i in issues if i["severity"] == "major")
    minor = sum(1 for i in issues if i["severity"] == "minor")

    score = 100
    for i in issues:
        if i["type"] in ("spelling", "grammar"):
            score -= 10 if i["severity"] == "major" else 4
        elif i["type"] == "punctuation":
            score -= 2
        else:
            score -= 2
    score = int(clamp(score, 0, 100))

    unclear_ratio = (uncertain_count / max(1, total_words)) if total_words else 0.0
    detection_conf = int(clamp(ocr_confidence - unclear_ratio * 35 + 10, 45, 95))

    return score, ocr_confidence, detection_conf, {"total": len(issues), "major": major, "minor": minor}


def run_ocr(image: Image.Image) -> Tuple[List[str], List[Dict[str, Any]], int, str]:
    """OCR with Tesseract. Produces line-based text + [unclear] markers for low-confidence words."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))

    lines: Dict[Tuple[int, int, int], List[Tuple[int, str]]] = {}
    uncertain: List[Dict[str, Any]] = []
    confs: List[int] = []

    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf = safe_int(data["conf"][i], default=-1)
        if conf >= 0:
            confs.append(conf)
        if not text:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, [])

        if conf >= 0 and conf < OCR_UNCLEAR_CONF_THRESHOLD:
            uncertain.append({"line": None, "text": text, "note": f"low OCR confidence ({conf})"})
            lines[key].append((data["word_num"][i], "[unclear]"))
        else:
            lines[key].append((data["word_num"][i], text))

    ordered_keys = sorted(lines.keys(), key=lambda k: (k[0], k[1], k[2]))
    extracted_lines: List[str] = []
    for key in ordered_keys:
        words = sorted(lines[key], key=lambda t: t[0])
        line_text = " ".join(w for _, w in words).strip()
        if line_text:
            extracted_lines.append(line_text)

    for u in uncertain:
        token = u["text"]
        found = None
        for idx, ln in enumerate(extracted_lines, start=1):
            if "[unclear]" in ln or token in ln:
                found = idx
                break
        u["line"] = found or 1

    avg_conf = int(round(sum(confs) / len(confs))) if confs else 0
    ocr_confidence = int(clamp(avg_conf, 0, 100))

    raw_text = "\n".join(extracted_lines)
    return extracted_lines, uncertain, ocr_confidence, raw_text


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    variant: Variant = Form("us"),
):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    extracted_lines, uncertain, ocr_confidence, text = run_ocr(image)

    if not extracted_lines:
        return {
            "language_variant": variant,
            "accuracy_score": 0,
            "ocr_confidence": 0,
            "detection_confidence": 0,
            "counts": {"total": 0, "major": 0, "minor": 0},
            "extracted_lines": [],
            "uncertain": [],
            "issues": [],
            "corrected_text": "",
        }

    tool = language_tool_python.LanguageTool(variant_to_lt_language(variant))
    matches = tool.check(text)

    issues: List[Dict[str, Any]] = []
    for m in matches:
        issue_type, severity = classify_issue(m)

        offset = int(m.offset)
        length = int(m.errorLength)
        original = text[offset : offset + length] if length > 0 else ""

        repls = list(getattr(m, "replacements", []) or [])
        fix = repls[0] if repls else ""
        why = getattr(m, "message", "") or ""

        # If BOTH, don't treat US/UK variants as errors.
        if variant == "both" and original and is_probable_us_uk_variant(original):
            continue

        line_no = line_number_from_offset(text, offset)

        issues.append(
            {
                "type": issue_type,
                "severity": severity,
                "line": line_no,
                "original": original,
                "why": why,
                "fix": fix,
                "replacements": repls[:MAX_REPLACEMENTS],
            }
        )

    corrected = apply_fixes(text, matches)
    total_words = len([w for w in re.split(r"\s+", text) if w])

    accuracy_score, ocr_conf, detection_conf, counts = compute_scores(
        issues=issues,
        ocr_confidence=ocr_confidence,
        uncertain_count=len(uncertain),
        total_words=total_words,
    )

    return {
        "language_variant": variant,
        "accuracy_score": accuracy_score,
        "ocr_confidence": ocr_conf,
        "detection_confidence": detection_conf,
        "counts": {"total": counts["total"], "major": counts["major"], "minor": counts["minor"]},
        "extracted_lines": extracted_lines,
        "uncertain": uncertain,
        "issues": [
            {
                "type": i["type"],
                "severity": i["severity"],
                "line": i["line"],
                "original": i["original"],
                "why": i["why"],
                "fix": i["fix"],
            }
            for i in issues
        ],
        "corrected_text": corrected,
    }

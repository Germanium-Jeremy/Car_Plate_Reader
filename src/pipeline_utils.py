import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract

MIN_AREA = 600
AR_MIN = 2.0
AR_MAX = 8.0
W_OUT = 450
H_OUT = 140
OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DEFAULT_PLATE_RE = re.compile(r"[A-Z]{3}[0-9]{3}[A-Z]")


def detect_plate_candidates(
    frame: np.ndarray,
    min_area: int = MIN_AREA,
    ar_min: float = AR_MIN,
    ar_max: float = AR_MAX,
) -> list:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect
        if w <= 0 or h <= 0:
            continue

        ar = max(w, h) / max(1.0, min(w, h))
        if ar_min <= ar <= ar_max:
            candidates.append(rect)

    candidates.sort(key=lambda r: r[1][0] * r[1][1], reverse=True)
    return candidates


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def warp_plate(frame: np.ndarray, rect: tuple, width: int = W_OUT, height: int = H_OUT) -> np.ndarray:
    box = cv2.boxPoints(rect)
    src = order_points(box)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (width, height))


def draw_rotated_rect(frame: np.ndarray, rect: tuple, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    box = cv2.boxPoints(rect).astype(int)
    cv2.polylines(frame, [box], True, color, thickness)
    return box


def preprocess_for_ocr(plate_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inverted = cv2.bitwise_not(thresh)
    return thresh, inverted


def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _score_ocr_text(text: str) -> int:
    score = len(text)
    if re.search(r"[A-Z]", text):
        score += 1
    if re.search(r"[0-9]", text):
        score += 1
    if DEFAULT_PLATE_RE.search(text):
        score += 3
    return score


def read_plate_text(plate_img: np.ndarray) -> tuple[str, np.ndarray]:
    thresh, inverted = preprocess_for_ocr(plate_img)
    variants = [thresh, inverted]
    configs = [
        f"--psm 7 --oem 3 -c tessedit_char_whitelist={OCR_WHITELIST}",
        f"--psm 8 --oem 3 -c tessedit_char_whitelist={OCR_WHITELIST}",
    ]

    best_text = ""
    best_score = -1
    best_img = thresh

    for image in variants:
        for config in configs:
            raw = pytesseract.image_to_string(image, config=config)
            cleaned = clean_ocr_text(raw)
            score = _score_ocr_text(cleaned)
            if score > best_score:
                best_score = score
                best_text = cleaned
                best_img = image

    return best_text, best_img


def extract_valid_plate(text: str, pattern=DEFAULT_PLATE_RE) -> str | None:
    text = clean_ocr_text(text)
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    match = pattern.search(text)
    if match:
        return match.group(0)
    return None


def majority_vote(buffer: list[str]) -> str | None:
    if not buffer:
        return None
    values, counts = np.unique(np.array(buffer), return_counts=True)
    return str(values[np.argmax(counts)])


def append_plate_log(csv_path: str, plate: str, ts: str | None = None) -> None:
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    if ts is None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

    write_header = not csv_file.exists()
    with open(csv_file, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["Plate Number", "Timestamp"])
        writer.writerow([plate, ts])

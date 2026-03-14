import argparse
import os
import time
from collections import deque

import cv2
import numpy as np

from pipeline_utils import (
    append_plate_log,
    detect_plate_candidates,
    draw_rotated_rect,
    extract_valid_plate,
    majority_vote,
    read_plate_text,
    warp_plate,
)

PLATE_PATTERN = r"[A-Z]{3}[0-9]{3}[A-Z]"
BUFFER_SIZE = 5
COOLDOWN_SECONDS = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Live extraction with temporal confirmation and CSV logging")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--buffer", type=int, default=BUFFER_SIZE, help="Frames required for confirmation")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_SECONDS, help="Seconds before saving same plate again")
    return parser.parse_args()


def log_path() -> str:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_dir, "data", "logs", "plates_log.csv")


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened")

    csv_path = log_path()
    plate_buffer = deque(maxlen=max(2, args.buffer))
    last_saved_plate = None
    last_saved_time = 0.0
    aligned_plate = None
    ocr_input = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        candidates = detect_plate_candidates(frame)

        status = "Searching..."
        status_color = (0, 200, 255)

        if candidates:
            rect = candidates[0]
            box = draw_rotated_rect(vis, rect, color=(0, 255, 0), thickness=2)

            aligned_plate = warp_plate(frame, rect)
            raw_text, ocr_input = read_plate_text(aligned_plate)
            valid_plate = extract_valid_plate(raw_text, pattern=PLATE_PATTERN)

            status = "Detect -> Align -> OCR"
            status_color = (0, 255, 0)

            x = max(0, int(np.max(box[:, 0])) - 240)
            y = min(vis.shape[0] - 10, int(np.max(box[:, 1])) + 25)

            if valid_plate:
                plate_buffer.append(valid_plate)
                cv2.putText(vis, f"VALID: {valid_plate}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if len(plate_buffer) == plate_buffer.maxlen:
                    confirmed = majority_vote(list(plate_buffer))
                    now = time.time()
                    allow_save = (
                        confirmed is not None
                        and (
                            confirmed != last_saved_plate
                            or (now - last_saved_time) >= args.cooldown
                        )
                    )

                    cv2.putText(
                        vis,
                        f"CONFIRMED: {confirmed}",
                        (x, max(25, y - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 220, 0),
                        2,
                    )

                    if allow_save:
                        append_plate_log(csv_path, confirmed)
                        print(f"[SAVED] {confirmed}")
                        last_saved_plate = confirmed
                        last_saved_time = now

                    plate_buffer.clear()
            elif raw_text:
                cv2.putText(vis, f"RAW: {raw_text}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

        cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(vis, f"Buffer: {len(plate_buffer)}/{plate_buffer.maxlen}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "Press q to quit", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Temporal Pipeline", vis)

        if aligned_plate is not None:
            cv2.imshow("Aligned Plate", aligned_plate)
        if ocr_input is not None:
            cv2.imshow("OCR Preprocess", ocr_input)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

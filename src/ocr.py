import argparse

import cv2
import numpy as np

from pipeline_utils import detect_plate_candidates, draw_rotated_rect, read_plate_text, warp_plate


def parse_args():
    parser = argparse.ArgumentParser(description="OCR on aligned plate region")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        candidates = detect_plate_candidates(frame)

        msg = "Searching for plate..."
        color = (0, 200, 255)
        plate_img = None
        ocr_input = None

        if candidates:
            rect = candidates[0]
            box = draw_rotated_rect(vis, rect, color=(0, 255, 0), thickness=2)

            plate_img = warp_plate(frame, rect)
            plate_text, ocr_input = read_plate_text(plate_img)

            msg = "OCR running"
            color = (0, 255, 0)

            if plate_text:
                x = max(0, int(np.max(box[:, 0])) - 180)
                y = min(vis.shape[0] - 10, int(np.max(box[:, 1])) + 25)
                cv2.putText(vis, f"OCR: {plate_text}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, "Press q to quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("OCR Stage", vis)

        if plate_img is not None:
            cv2.imshow("Aligned Plate", plate_img)
        if ocr_input is not None:
            cv2.imshow("OCR Preprocess", ocr_input)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

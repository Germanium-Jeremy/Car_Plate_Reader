import argparse

import cv2

from pipeline_utils import detect_plate_candidates, draw_rotated_rect


def parse_args():
    parser = argparse.ArgumentParser(description="Plate candidate detection")
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

        if candidates:
            msg = f"Plate candidate(s): {len(candidates)}"
            color = (0, 255, 0)
            for idx, rect in enumerate(candidates[:3]):
                draw_rotated_rect(
                    vis,
                    rect,
                    color=(0, 255, 0) if idx == 0 else (255, 255, 0),
                    thickness=2,
                )

        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, "Press q to quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Plate Detection", vis)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import argparse

import cv2

from pipeline_utils import detect_plate_candidates, draw_rotated_rect, warp_plate


def parse_args():
    parser = argparse.ArgumentParser(description="Plate alignment and rectification")
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
        aligned_plate = None

        msg = "Detecting plate..."
        color = (0, 200, 255)

        if candidates:
            best_rect = candidates[0]
            draw_rotated_rect(vis, best_rect, color=(255, 0, 0), thickness=2)
            aligned_plate = warp_plate(frame, best_rect)
            msg = "Plate aligned (450x140)"
            color = (0, 255, 0)

        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, "Press q to quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Alignment Stage", vis)

        if aligned_plate is not None:
            cv2.imshow("Aligned Plate", aligned_plate)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

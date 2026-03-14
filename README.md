## Car Plate Reader (Classical OpenCV + Tesseract)

This project follows the **Car Number Plate Extraction in Three Steps** book:

1. Plate Detection
2. Plate Alignment
3. OCR

Then two practical stages:

4. Regex Validation
5. Temporal Confirmation + CSV Logging

## Dependencies

- Python 3
- `opencv-python`
- `numpy`
- `pytesseract`
- Tesseract OCR binary installed on your OS

Install packages:

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python numpy pytesseract
```

## Run Stages

From project root:

1. Testing if the camera works

```bash
python src/camera.py --camera 0
```
2. Detecting the car plate
```bash
python src/detect.py --camera 0
```
3. Aligning the detected plate horizontally
```bash
python src/align.py --camera 0
```
4. Extracting plate number
```bash
python src/ocr.py --camera 0
```
5. Validating the extracted plate number
```bash
python src/validate.py --camera 0
python src/temporal.py --camera 0
```

If your webcam is on another index, replace `--camera 0` with `--camera 1` (or another value).

## Output

Confirmed plate reads are saved to:

`data/logs/plates_log.csv`

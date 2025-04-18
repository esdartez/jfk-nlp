import zipfile
import io
import os
from pathlib import Path
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from status_db import StatusDB

# Settings
ZIP_DIR = Path("data/")
OUTPUT_DIR = Path("output/text/")
DB_PATH = "ocr_status.sqlite"
DPI = 200
MAX_THREADS = 4

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_ocr_text(raw_text: str) -> str:
    lines = raw_text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Comment these out for debugging:
        # if line.startswith("RELEASE UNDER") or "DocId:" in line:
        #     continue
        # if line.startswith("--- Page") or line.startswith("HW"):
        #     continue
        cleaned.append(line)
    return "\n".join(cleaned)

def process_pdf(zip_path, pdf_name, pdf_bytes, db: StatusDB):
    doc_id = f"{zip_path.stem}::{pdf_name}"
    if db.get_status(doc_id) == "success":
        return

    try:
        images = convert_from_bytes(pdf_bytes, dpi=DPI)
        full_text = []

        for i, img in enumerate(images):
            img_gray = img.convert("L")
            text = pytesseract.image_to_string(img_gray)
            full_text.append(f"\n--- Page {i+1} ---\n{text.strip()}")

        output_file = OUTPUT_DIR / f"{zip_path.stem}__{pdf_name.replace('/', '_')}.txt"
        clean_text = clean_ocr_text("\n".join(full_text))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(clean_text)

        db.set_status(doc_id, "success")

    except Exception as e:
        db.set_status(doc_id, "failed", notes=str(e))

def process_zip(zip_path, db: StatusDB):
    with zipfile.ZipFile(zip_path, "r") as z:
        pdf_files = [f for f in z.namelist() if f.lower().endswith(".pdf")]

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []
            for pdf_name in pdf_files:
                with z.open(pdf_name) as file:
                    pdf_bytes = file.read()
                    futures.append(executor.submit(process_pdf, zip_path, pdf_name, pdf_bytes, db))

            for f in tqdm(futures, desc=f"Processing {zip_path.name}"):
                f.result()

def main():
    db = StatusDB(DB_PATH)

    # Test a single ZIP file
    zip_path = ZIP_DIR / "test_batch1.zip"
    process_zip(zip_path, db)

    db.close()

if __name__ == "__main__":
    main()

# JFK-NLP

**JFK-NLP** is a scalable pipeline for performing OCR and natural language processing on the publicly released JFK assassination documents. It is designed to convert large quantities of scanned government records into machine-readable text and analyze them using standard NLP techniques.

This project was developed as part of a final assignment for an NLP course.

## Project Structure

```
jfk-nlp/
├── scripts/        # Python modules (OCR, NLP, status tracking)
├── data/           # Input ZIP files containing JFK PDFs
├── output/         # OCR and NLP results (TXT, CSV, JSON)
├── logs/           # Log files for errors and retries
├── report/         # Milestone and final project reports
├── .gitignore      # Git exclusions
└── README.md       # This file
```

## Features

- OCR using Tesseract and `pytesseract`
- In-memory ZIP + PDF processing with `pdf2image`
- Parallelized image-to-text processing with progress tracking
- Resume-safe pipeline using SQLite to avoid redundant processing
- Basic NLP: word frequency analysis, bigram extraction, lexical richness
- Named Entity Recognition using spaCy
- Structured CSV export of all summary data

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/jfk-nlp.git
cd jfk-nlp
```

### 2. Create Conda Environment

```bash
conda create -n jfk-nlp python=3.10 -y
conda activate jfk-nlp
```

### 3. Install System Dependencies (Ubuntu/WSL)

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils -y
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
```

## Planned Enhancements

- Fallback OCR for handwritten text using TrOCR
- Automated OCR quality scoring and filtering
- Full-topic modeling and semantic clustering
- Retrieval-based question answering (QA) interface
- FAISS-based semantic search over document embeddings

## License

This project is intended for educational and research use only.

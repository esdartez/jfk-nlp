import difflib
import string
from pathlib import Path
from nltk.corpus import words
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

nltk.download("punkt")
nltk.download("words")

ENGLISH_WORDS = set(words.words())

# Load GPT-2 model and tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

def load_text(path):
    return Path(path).read_text(encoding='utf-8').strip()

def word_accuracy(ocr_text, reference_text):
    ocr_tokens = word_tokenize(ocr_text)
    ref_tokens = word_tokenize(reference_text)
    matcher = difflib.SequenceMatcher(None, ocr_tokens, ref_tokens)
    return matcher.ratio()

def valid_word_ratio(text):
    tokens = word_tokenize(text)
    valid = [t for t in tokens if t.lower() in ENGLISH_WORDS]
    return len(valid) / len(tokens) if tokens else 0

def perplexity_score(text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings.input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()

def evaluate_ocr(ocr_path, reference_path=None):
    ocr_text = load_text(ocr_path)
    results = {
        "file": ocr_path.name,
        "length_chars": len(ocr_text),
        "valid_word_ratio": round(valid_word_ratio(ocr_text), 4),
        "perplexity_score": None,
        "word_accuracy_vs_reference": None
    }

    try:
        results["perplexity_score"] = round(perplexity_score(ocr_text), 4)
    except Exception as e:
        results["perplexity_score"] = -1  # -1 indicates failure to compute

    if reference_path:
        reference_text = load_text(reference_path)
        results["word_accuracy_vs_reference"] = round(
            word_accuracy(ocr_text, reference_text), 4
        )

    return results

def batch_evaluate(ocr_dir, ref_dir=None, out_csv="ocr_eval_results.csv"):
    ocr_dir = Path(ocr_dir)
    ref_dir = Path(ref_dir) if ref_dir else None
    records = []

    for ocr_file in ocr_dir.glob("*.txt"):
        ref_file = ref_dir / ocr_file.name if ref_dir else None
        record = evaluate_ocr(ocr_file, ref_file if ref_file and ref_file.exists() else None)
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Evaluation results saved to {out_csv}")

if __name__ == "__main__":
    # Example usage
    batch_evaluate("output/text", ref_dir=None)
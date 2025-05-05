import os
import re
import argparse
from collections import Counter
from itertools import islice
import spacy
import nltk
from nltk import word_tokenize, bigrams
from nltk.corpus import stopwords
from textblob import TextBlob
from transformers import pipeline

nltk.download('punkt')
nltk.download('stopwords')

# Load models
nlp_ner = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Utility
def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

# 1. Named Entity Recognition
def extract_ner(text):
    doc = nlp_ner(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# 2. Word frequency + bigrams
def extract_word_stats(text, top_n=20):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words("english")]
    freq = Counter(tokens)
    bigram_freq = Counter(bigrams(tokens))
    return freq.most_common(top_n), bigram_freq.most_common(top_n)

# 3. Sentiment
def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# 4. Summarization
def summarize_text(text, max_tokens=512):
    if len(text.split()) > max_tokens:
        text = ' '.join(text.split()[:max_tokens])
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

# --- Main Experiment Driver ---
def run_nlp_analysis(filepath):
    raw_text = clean_text(load_text(filepath))
    print(f"\nLoaded file: {filepath}\n")

    # Named Entities
    print("Named Entities:")
    entities = extract_ner(raw_text)
    for text, label in entities[:10]:
        print(f"  [{label}]: {text}")

    # Frequencies
    print("\n Top Words & Bigrams:")
    top_words, top_bigrams = extract_word_stats(raw_text)
    print("  Top Words:", top_words)
    print("  Top Bigrams:", [' '.join(bg) for bg, _ in top_bigrams])

    # Sentiment
    polarity, subjectivity = sentiment_analysis(raw_text)
    print(f"\n Sentiment:\n  Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}")

    # Summary
    print("\n Summary:")
    print(" ", summarize_text(raw_text))

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to cleaned OCR text file")
    args = parser.parse_args()

    run_nlp_analysis(args.file)

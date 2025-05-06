import os
import spacy
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate


# 1. Load OCR-cleaned text files

docs = []
data_dir = "/home/edartez/jfk-nlp/JFK_Records/texts"  # <- Adjust path if needed

for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(data_dir, filename))
        docs.extend(loader.load())

# 2. Chunk documents for retrieval

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed and index with FAISS

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)


# 4. Load Falcon-7B model

model_id = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,  # Will be deprecated—migrate to quantization_config soon
    trust_remote_code=True  # Will also be deprecated
)

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id
)


# 5. Wrap in LangChain interface

llm = HuggingFacePipeline(pipeline=qa_pipeline)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# 6. NER-based query enrichment

nlp_ner = spacy.load("en_core_web_sm")
RELEVANT_LABELS = {"PERSON", "ORG", "GPE", "DATE", "NORP", "EVENT", "FAC"}

def enrich_query_with_entities(query: str) -> str:
    doc = nlp_ner(query)
    ents = [ent.text for ent in doc.ents if ent.label_ in RELEVANT_LABELS]
    if ents:
        enriched = query + " | Entities: " + ", ".join(ents)
        return enriched
    return query

# 7. Ask a real question

query = "What does the Warren Commission conclude about Lee Harvey Oswald?"
enriched_query = enrich_query_with_entities(query)

# Prompt to reduce hallucination
final_prompt = f"""Use the following JFK archive excerpts to answer the question.

If the answer is unknown or unclear, say "The documents do not provide a definitive answer."

Context: {enriched_query}

Question: {query}
Answer:"""

print("\nENRICHED QUERY:", enriched_query)
response = qa_chain.invoke(final_prompt)

print("\nQUESTION:", query)
print("ANSWER:", response['result'])


# 8. ROUGE + SQuAD Evaluation

reference = "The Warren Commission concluded that Lee Harvey Oswald acted alone."
prediction = response['result']

# ROUGE
rouge = evaluate.load("rouge")
scores = rouge.compute(predictions=[prediction], references=[reference])
print("\nROUGE SCORES:")
for k, v in scores.items():
    print(f"{k}: {v:.3f}")

# SQuAD
squad = evaluate.load("squad")
squad_preds = [{"id": "1", "prediction_text": prediction}]
squad_refs = [{"id": "1", "answers": {"text": [reference], "answer_start": [0]}}]
squad_scores = squad.compute(predictions=squad_preds, references=squad_refs)

print("\nSQuAD SCORES:")
print(f"Exact Match: {squad_scores['exact_match']:.2f}")
print(f"Token F1:     {squad_scores['f1']:.2f}")

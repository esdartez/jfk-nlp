import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Load all .txt files from the JFK corpus directory
docs = []
data_dir = "/home/edartez/jfk-nlp/JFK_Records/texts"  # <- Change this if needed

for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(data_dir, filename))
        docs.extend(loader.load())

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Generate embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

model_id = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
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


# 5. Wrap model in LangChain interface
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# 6. Set up Retrieval-Augmented Generation
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# 7. Ask a real question
query = "What does the Warren Commission conclude about Lee Harvey Oswald?"
response = qa_chain.invoke(query)

# 8. Print answer
print("\nQUESTION:", query)
print("ANSWER:", response['result'])


def compute_rouge_evaluate(predictions, references):
    rouge = evaluate.load('rouge')
    # returns dict with 'rouge1', 'rouge2', 'rougeL', each fmeasure, precision, recall
    return rouge.compute(predictions=predictions, references=references)

prediction = response['result']
reference  = "The Warren Commission concluded that Lee Harvey Oswald acted alone."

scores = compute_rouge_evaluate([prediction], [reference])
print("ROUGE scores (evaluate lib):")
for k, v in scores.items():
    print(f"{k}: {v:.3f}")
squad = evaluate.load("squad")

# `predictions` should be a list of dicts: [{"id": str, "prediction_text": str}, …]
# `references`   a list of dicts: [{"id": str, "answers": {"text": [str,…]}}, …]
preds = [{"id": "1", "prediction_text": response['result']}]
refs  = [{"id": "1", "answers": {"text": [reference]}}]

metrics = squad.compute(predictions=preds, references=refs)
print(f"Exact Match: {metrics['exact_match']}")  
print(f"Token-F1:      {metrics['f1']}")
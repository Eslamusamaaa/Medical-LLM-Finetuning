import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

input_file = r"C:\Users\Lenovo\Medical-LLM-Finetuning\deployment\BackEnd\qwen_ready_medical_qa(2).jsonl"

print("Loading documents from JSONL...")
documents = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            
            question = data.get("instruction", "") + " " + data.get("input", "")
            answer = data.get("output", "")
            
            if not question.strip() or not answer.strip():
                continue
                
            doc_text = f"medical question: {question.strip()}\n medical answer: {answer.strip()}"
            documents.append(Document(page_content=doc_text))
            
        except json.JSONDecodeError:
            continue

print(f"Loaded {len(documents)} questions and answers.")

print("downloading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("building FAISS index...0")
vector_db = FAISS.from_documents(documents, embeddings)

vector_db.save_local("faiss_index")
print("saved to 'faiss_index'!")
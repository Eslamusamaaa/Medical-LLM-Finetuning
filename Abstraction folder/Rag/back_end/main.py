import time, sys, os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# استخدام المسار المباشر (Absolute Path) للواجهة لضمان عدم حدوث أي خطأ
FRONTEND_DIR = Path(r"C:\Users\Laptop World\Desktop\Abstract\Rag\Front_end")

REPO_ID = "hagora-30/qwen2.5-1.5B-medical-arabic"

SYSTEM_PROMPT = """أنت مساعد طبي ذكي وموثوق.
مهمتك الإجابة على سؤال المريض بالاعتماد على "المعلومات المرجعية" المرفقة فقط.
أجب باللغة العربية الفصحى فقط. يُمنع منعاً باتاً استخدام اللغة الصينية أو الإنجليزية.
قم بصياغة الإجابة في نقاط واضحة ومنسقة لتسهيل القراءة."""

tokenizer = None
model = None
vector_db = None  
embeddings_model = None

def load_ai_components():
    global tokenizer, model, vector_db
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Setting AI components to use: {device.upper()}")
    
    faiss_path = r"C:\Users\Laptop World\Desktop\Abstract\Rag"
    logger.info(f"Loading FAISS Vector Database from: {faiss_path} ...")   
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': device}
    )

    vector_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    logger.info("Vector Database loaded successfully!")

    logger.info(f"Loading Qwen model from: {REPO_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        torch_dtype="auto",
        device_map="auto" 
    )
    logger.info("Qwen Model ready on GPU!")

def generate(question: str, max_tokens: int = 512, temperature: float = 0.2) -> str:

    retrieved_docs = vector_db.similarity_search(question, k=2)
    
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    augmented_user_prompt = f"""
المعلومات المرجعية:
{context_text}

سؤال المريض:
{question}
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": augmented_user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature else 0.3,
        do_sample=True ,
        repetition_penalty=1.1
    )
    
    response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ai_components()
    logger.info("Server ready at http://localhost:8000")
    yield
    logger.info("Shutting down.")

app = FastAPI(title="Arabic Medical AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
else:
    logger.error(f"Frontend directory STILL NOT FOUND at {FRONTEND_DIR}. Please check the path!")

class ChatRequest(BaseModel):
    message:     str   = Field(..., min_length=2, max_length=2000)
    max_tokens:  int   = Field(default=512, ge=64, le=1024)
    temperature: float = Field(default=0.3, ge=0.1, le=1.5)

class ChatResponse(BaseModel):
    response:        str
    elapsed_seconds: float
    device:          str

@app.get("/")
async def serve_ui():
    if not (FRONTEND_DIR / "index.html").exists():
        return {"message": f"Frontend index.html not found in {FRONTEND_DIR}"}
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty.")
    
    start = time.time()
    try:
        response_text = generate(req.message, req.max_tokens, req.temperature)
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    return ChatResponse(
        response=response_text,
        elapsed_seconds=round(time.time() - start, 2),
        device=str(model.device)
    )

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": REPO_ID}
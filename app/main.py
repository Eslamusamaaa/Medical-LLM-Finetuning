"""FastAPI server for the Arabic medical assistant (Qwen2.5-1.5B fine-tuned on Arabic medical QA).

    uvicorn app.main:app --host 127.0.0.1 --port 8000        # from the project root
    USE_RAG=0 uvicorn app.main:app ...                       # answer from the model only (no retrieval)

With RAG (default) the two most similar question/answer pairs from rag_index/ are added to the prompt.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "app" / "static"
RAG_DIR = ROOT / "rag_index"
REPO_ID = os.environ.get("MODEL_ID", "hagora-30/qwen2.5-1.5B-medical-arabic")
USE_RAG = os.environ.get("USE_RAG", "1") != "0"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# the same system prompt the model was fine-tuned with (src/finetune.py), so safety behaviour is kept with RAG too
SYSTEM_PROMPT = (
    "أنت مساعد طبي ذكي وموثوق. تقدم معلومات وإرشادات طبية عامة باللغة العربية السليمة، "
    "وتوجه المريض دائماً لاستشارة الطبيب في الحالات الحرجة. "
    "لا تقم بوصف جرعات محددة."
)
RAG_INSTRUCTION = (
    "اعتمد على \"المعلومات المرجعية\" المرفقة عند الإجابة، ولا تذكر معلومات غير موجودة فيها إذا لم تكن متأكداً. "
    "أجب باللغة العربية فقط وبنقاط واضحة ومنسقة."
)
DISCLAIMER = "هذه معلومات عامة للتثقيف الصحي وليست تشخيصاً أو بديلاً عن استشارة الطبيب."

tokenizer = model = vector_db = None


def load_components():
    global tokenizer, model, vector_db
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s | RAG: %s", device.upper(), USE_RAG)

    if USE_RAG:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": device})
        # the pickle is our own index built by src/build_vector_db.py; never load an index from an untrusted source
        vector_db = FAISS.load_local(str(RAG_DIR), embeddings, allow_dangerous_deserialization=True)
        logger.info("Vector database loaded from %s", RAG_DIR)

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, torch_dtype="auto", device_map="auto")
    logger.info("Model %s ready", REPO_ID)


def build_messages(question):
    if not USE_RAG:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    context = "\n\n".join(doc.page_content for doc in vector_db.similarity_search(question, k=2))
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n{RAG_INSTRUCTION}"},
        {"role": "user", "content": f"المعلومات المرجعية:\n{context}\n\nسؤال المريض:\n{question}"},
    ]


def generate(question, max_tokens=512, temperature=0.3):
    text = tokenizer.apply_chat_template(build_messages(question), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=True,
                            repetition_penalty=1.1)
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_components()
    yield


app = FastAPI(title="Arabic Medical AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=2000)
    max_tokens: int = Field(default=512, ge=64, le=1024)
    temperature: float = Field(default=0.3, ge=0.1, le=1.5)


class ChatResponse(BaseModel):
    response: str
    elapsed_seconds: float
    device: str
    rag: bool
    disclaimer: str = DISCLAIMER


@app.get("/")
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty.")
    start = time.time()
    try:
        answer = generate(req.message, req.max_tokens, req.temperature)
    except Exception:
        logger.exception("Generation error")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    return ChatResponse(response=answer, elapsed_seconds=round(time.time() - start, 2),
                        device=str(model.device), rag=USE_RAG)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": REPO_ID, "rag": USE_RAG}

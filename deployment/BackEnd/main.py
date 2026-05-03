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

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "FrontEnd"

REPO_ID = "hagora-30/qwen2.5-1.5B-medical-arabic"

SYSTEM_PROMPT =  """أنت مساعد طبي ذكي وموثوق. تقدم معلومات وإرشادات طبية عامة باللغة العربية السليمة، 
    وتوجه المريض دائماً لاستشارة الطبيب في الحالات الحرجة. 
    لا تقم بوصف جرعات محددة."""

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is not None:
        return

    logger.info(f"Loading model from Hugging Face: {REPO_ID}...")
    
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        torch_dtype="auto",
        device_map="auto"
    )
    logger.info("Model ready and loaded into GPU!")

def generate(question: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature else 0.3,
        do_sample=True
    )
    
    response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
    
    response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
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
        return {"message": "Frontend index.html not found"}
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
import time, sys, os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from llama_cpp import Llama
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent.parent
MODEL_PATH   = BASE_DIR / "models" / "7b_medical_model.gguf" 
FRONTEND_DIR = BASE_DIR / "FrontEnd"

SYSTEM_PROMPT = """أنت مساعد معلومات طبية حذر.

المهام:
* تقديم معلومات صحية عامة فقط.
* شرح الأعراض والأسباب المحتملة بشكل مبسط.
* تقديم نصائح عامة غير علاجية.

القواعد:
* لا تقم بالتشخيص أو تأكيد أي مرض.
* لا تصف أدوية أو جرعات.
* لا تقدم قرارات علاجية.
* إذا كان السؤال يتعلق بحالة طارئة أو خطيرة، انصح المستخدم بالتوجه إلى طبيب أو طوارئ فورًا.
* إذا لم تكن متأكدًا من المعلومة، اذكر بوضوح أنك غير متأكد.

أسلوب الإجابة:
* استخدم لغة بسيطة وواضحة.
* تجنب المصطلحات الطبية المعقدة قدر الإمكان.
* لا تستخدم نبرة حاسمة أو مؤكدة.
* لا تخترع معلومات."""

_model     = None

def load_model():
    global _model
    if _model is not None:
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    logger.info("Loading model on GPU via llama.cpp...")

    
    _model = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,           
        n_gpu_layers=0,      
        verbose=False         
    )
    logger.info(" Model ready!")

medical_prompt = """أنت مساعد طبي ذكي وموثوق باللغة العربية. دورك الأساسي هو تقديم معلومات طبية عامة، وتثقيف صحي بأسلوب علمي، دقيق، ومبسط.
يجب أن تكون إجابتك مفصلة، ومنسقة في نقاط مرتبة (Bullet points) لتسهيل القراءة.
أجب على السؤال الطبي التالي بدقة:

### السؤال:
{}

### الإجابة:
"""

def generate(question: str, max_tokens: int = 512, temperature: float = 0.4) -> str:
    formatted_prompt = medical_prompt.format(question)
    
    
    response = _model(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.2,
        stop=["###", "السؤال:", "<|endoftext|>", "<|im_end|>"] 
    )
    
    return response["choices"][0]["text"].strip()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info(" Server ready at http://localhost:8000")
    yield
    logger.info(" Shutting down.")

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
        raise HTTPException(status_code=500, detail=str(e))
    
    return ChatResponse(
        response=response_text,
        elapsed_seconds=round(time.time() - start, 2),
        device="GPU (llama.cpp)"
    )

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "qwen-2.5-7B-medical-gguf"}
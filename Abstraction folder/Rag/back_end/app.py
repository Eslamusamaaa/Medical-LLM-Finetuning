import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
# ... (any other imports you had for your model, like transformers, torch, etc.)

# 1. Initialize the app BEFORE any routes
app = FastAPI()

# 2. Define your directories and models here
# (Make sure FRONTEND_DIR, REPO_ID, model, generate(), logger, etc., are defined here)
# FRONTEND_DIR = ...
# REPO_ID = ...
# model = ...

# 3. Request/Response Models
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    elapsed_seconds: float
    device: str

# 4. Your Routes
@app.get("/")
async def serve_ui():
    # Make sure FRONTEND_DIR is defined above!
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
        # Make sure 'logger' is defined above, or just use print(f"Generation error: {str(e)}")
        print(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    return ChatResponse(
        response=response_text,
        elapsed_seconds=round(time.time() - start, 2),
        device=str(model.device)
    )

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": REPO_ID}
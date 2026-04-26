import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

# Mount the static folder to serve the HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    question: str
    max_tokens: int = 150

# Global flags and variables
MODEL_LOADED = False
model = None
tokenizer = None

@app.on_event("startup")
def load_model():
    global model, tokenizer, MODEL_LOADED
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # This is the folder you will put in ANLP later
        model_path = "./Medical_Qwen_Final" 
        
        if os.path.exists(model_path):
            print("Loading the real trained model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            MODEL_LOADED = True
            print("Model loaded successfully!")
        else:
            print(f"Model folder '{model_path}' not found. Running in Dummy Mode.")
            
    except ImportError:
        print("PyTorch or Transformers not installed. Running in Dummy Mode.")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    global MODEL_LOADED, model, tokenizer
    
    if not MODEL_LOADED:
        # Fallback response so you can test the UI right now
        dummy_answer = f"هذا رد تجريبي من السيرفر. سؤالك كان: '{request.question}'. الموديل الحقيقي لم يتم تحميله بعد!"
        return {"answer": dummy_answer}
        
    import torch
    prompt = f"User: {request.question}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=request.max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output.split("Assistant:")[-1].strip()
    
    return {"answer": answer}
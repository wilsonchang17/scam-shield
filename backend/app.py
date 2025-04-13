from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
import os
from ml_classifier import classify_message_ml
import numpy as np

app = FastAPI(
    title="Scam Shield API",
    description="An API for detecting scam messages using machine learning",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define base directory
static_dir = os.path.join(os.path.dirname(__file__), "static")

# Class for API request
class Message(BaseModel):
    text: str

@app.post("/classify")
def classify_message(msg: Message):
    try:
        print(f"Processing message: {msg.text}")
        ml_result = classify_message_ml(msg.text)
        
        # Convert NumPy types to Python native types
        if isinstance(ml_result, np.integer):
            ml_result = int(ml_result)
        
        # Convert numerical result to string label
        label = "Scam" if ml_result == 1 else "Not Scam"
        
        print(f"Classification result: {label}")
        
        # Return proper JSON response
        return {"label": label, "message": msg.text}
    except Exception as e:
        print(f"Error in classification: {e}")
        return JSONResponse({"label": "Error", "error": str(e)}, status_code=500)

# Root redirect to index
@app.get("/")
def root_redirect():
    return RedirectResponse(url="/index.html")

# Research redirect
@app.get("/research")
def research_redirect():
    return RedirectResponse(url="/research.html?v=1.1")

# Mount static files AFTER defining API routes
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import os
import uuid
from threading import Lock
from typing import Dict, Optional

from ml_classifier import classify_message_ml

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


class Message(BaseModel):
    text: str


pending_results: Dict[str, Dict[str, object]] = {}
pending_lock = Lock()


def _store_pending(task_id: str, status: str, payload: Optional[dict] = None) -> None:
    with pending_lock:
        pending_results[task_id] = {"status": status, "result": payload}


def _get_pending(task_id: str) -> Optional[Dict[str, object]]:
    with pending_lock:
        return pending_results.get(task_id)


def _remove_pending(task_id: str) -> Optional[Dict[str, object]]:
    with pending_lock:
        return pending_results.pop(task_id, None)


def _complete_classification(task_id: str, message_text: str) -> None:
    try:
        final_payload = classify_message_ml(message_text)
        _store_pending(task_id, "completed", final_payload)
    except Exception as exc:  # noqa: BLE001
        with pending_lock:
            pending_results[task_id] = {"status": "error", "error": str(exc)}


@app.post("/classify")
def classify_message(msg: Message, background_tasks: BackgroundTasks):
    try:
        print(f"Processing message: {msg.text}")
        result_payload = classify_message_ml(msg.text, defer=True)
        print(f"Initial classification payload: {result_payload}")

        label = result_payload.get("label")
        reason = result_payload.get("reason")
        votes = result_payload.get("votes", {})
        vote_summary = result_payload.get("vote_summary")
        consensus_status = result_payload.get("consensus_status")
        decision_source = result_payload.get("decision_source")
        processing_state = result_payload.get("processing_state", "completed")
        pending_reason = result_payload.get("pending_reason", False)
        pending_decision = result_payload.get("pending_decision", False)
        raw_prediction = result_payload.get("raw_prediction")

        if not label:
            if pending_decision:
                label = "Pending"
            elif pending_reason and raw_prediction == 1:
                label = "Scam"
            else:
                label = "Unknown"

        response = {
            "label": label,
            "message": msg.text,
            "reason": reason,
            "votes": votes,
            "vote_summary": vote_summary,
            "consensus_status": consensus_status,
            "decision_source": decision_source,
            "processing_state": processing_state,
            "pending_reason": pending_reason,
            "pending_decision": pending_decision,
            "raw_prediction": raw_prediction,
            "task_id": None,
        }

        if processing_state != "completed":
            task_id = str(uuid.uuid4())
            response["task_id"] = task_id
            response["status"] = processing_state
            _store_pending(task_id, "pending")
            background_tasks.add_task(_complete_classification, task_id, msg.text)

        return response
    except Exception as e:
        print(f"Error in classification: {e}")
        return JSONResponse({"label": "Error", "error": str(e)}, status_code=500)


@app.get("/classify/status/{task_id}")
def get_classification_status(task_id: str):
    record = _get_pending(task_id)
    if not record:
        return JSONResponse({"status": "not_found", "message": "Task ID not found"}, status_code=404)

    status = record.get("status", "pending")
    if status == "completed":
        _remove_pending(task_id)
        return {"status": "completed", "result": record.get("result")}
    if status == "error":
        _remove_pending(task_id)
        return {"status": "error", "error": record.get("error", "Unknown error")}
    return {"status": status}


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

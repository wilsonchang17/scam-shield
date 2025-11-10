import joblib
from collections import Counter
from typing import Any, Dict, Optional

# We will need the client in this file, so import it at the top.
from connectLLM import ScamShieldClient


# Load models
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
rf_model = joblib.load('models/random_forest_model.joblib')
dt_model = joblib.load('models/decision_tree_model.joblib')
xgb_model = joblib.load('models/xgboost_model.joblib')
knn_model = joblib.load('models/knn_model.joblib')

# Initialize the ScamShieldClient once
scam_shield_client = ScamShieldClient()


def _interpret_llm_label(label_value: Any) -> Optional[int]:
    """Normalize various LLM label responses into 1 (scam) or 0 (not scam)."""
    if label_value is None:
        return None

    if isinstance(label_value, (int, float)):
        if label_value >= 0.5:
            return 1
        if label_value <= 0:
            return 0
        return None

    if isinstance(label_value, str):
        normalized = label_value.strip().lower()
        if not normalized:
            return None

        scam_keywords = {
            "scam",
            "yes",
            "fraud",
            "malicious",
            "phishing",
            "likely scam",
            "very likely scam",
            "definitely scam",
        }
        safe_keywords = {
            "not scam",
            "legit",
            "legitimate",
            "ham",
            "safe",
            "no",
            "benign",
        }

        if normalized in scam_keywords or normalized.startswith("scam"):
            return 1
        if normalized in safe_keywords or normalized.startswith("not scam"):
            return 0
        if normalized in {"true", "1", "positive"}:
            return 1
        if normalized in {"false", "0", "negative"}:
            return 0

    return None


def classify_message_ml(message: str, defer: bool = False) -> Dict[str, Any]:
    print(f"ML votes: {message}")
    X = vectorizer.transform([message])

    preds = [
        rf_model.predict(X)[0],
        dt_model.predict(X)[0],
        xgb_model.predict(X)[0],
        knn_model.predict(X)[0],
    ]

    vote_count = Counter(preds)
    print(f"Vote count: {vote_count}")

    majority_vote, count = vote_count.most_common(1)[0]
    total_votes = len(preds)
    votes = {
        "scam": int(vote_count.get(1, 0)),
        "not_scam": int(vote_count.get(0, 0)),
    }
    vote_summary = f"Vote Scam: {votes['scam']} | Vote Not Scam: {votes['not_scam']}"

    consensus_status = ""
    decision_source = ""
    reason: Optional[str] = None
    final_label: Optional[int] = majority_vote
    processing_state = "completed"
    pending_reason = False
    pending_decision = False

    unanimous = count == total_votes

    if unanimous:
        decision_source = "ml_unanimous"
        consensus_label = "Scam" if majority_vote == 1 else "Not Scam"
        consensus_status = f"Unanimous {consensus_label} ({count}/{total_votes})"

        if majority_vote == 1:
            if defer:
                pending_reason = True
                processing_state = "pending_reason"
                reason = None
            else:
                print("ML models unanimously vote Scam. Getting explanation from LLM.")
                reason = scam_shield_client.explain_message(message)
        else:
            reason = None
    else:
        decision_source = "llm"
        consensus_status = f"LLM decision pending ({votes['scam']} vs {votes['not_scam']})"

        if defer:
            pending_decision = True
            processing_state = "pending_decision"
            final_label = None
        else:
            print("ML models disagree. Using LLM for prediction.")
            llm_result = scam_shield_client.predict_message(message)

            interpreted_label: Optional[int] = None
            llm_reason: Optional[str] = None

            if isinstance(llm_result, (list, tuple)) and len(llm_result) > 0:
                interpreted_label = _interpret_llm_label(llm_result[0])
                if len(llm_result) > 1:
                    llm_reason = llm_result[1]
            elif isinstance(llm_result, (str, int, float)):
                interpreted_label = _interpret_llm_label(llm_result)

            if interpreted_label is not None:
                final_label = interpreted_label
            else:
                print("LLM prediction format unexpected. Falling back to majority vote.")
                decision_source = "ml_fallback"
                final_label = majority_vote

            reason = llm_reason

            if final_label == 1 and reason is None:
                print("Fetching scam explanation from LLM for final decision.")
                reason = scam_shield_client.explain_message(message)

            label_for_status = "Scam" if final_label == 1 else "Not Scam"
            consensus_status = f"LLM decision: {label_for_status}"

    if final_label not in (0, 1) and final_label is not None:
        final_label = 1 if final_label else 0

    if isinstance(reason, (list, tuple)):
        reason = "\n".join(str(item) for item in reason if item)
    elif isinstance(reason, dict):
        reason_text = reason.get("explanation") or reason.get("reason")
        if isinstance(reason_text, (list, tuple)):
            reason = "\n".join(str(item) for item in reason_text if item)
        elif reason_text is not None:
            reason = str(reason_text)
        else:
            reason = str(reason)
    elif reason is not None and not isinstance(reason, str):
        reason = str(reason)

    label_text: Optional[str] = None
    if final_label in (0, 1):
        label_text = "Scam" if final_label == 1 else "Not Scam"
    elif unanimous and majority_vote == 1:
        label_text = "Scam"
    elif unanimous and majority_vote == 0:
        label_text = "Not Scam"

    return {
        "raw_prediction": int(final_label) if final_label is not None else None,
        "label": label_text,
        "reason": reason,
        "votes": votes,
        "vote_summary": vote_summary,
        "consensus_status": consensus_status,
        "decision_source": decision_source or "ml_majority",
        "processing_state": processing_state,
        "pending_reason": pending_reason,
        "pending_decision": pending_decision,
    }


if __name__ == '__main__':
    test_message = (
        "Virginia Department of Transportation Toll Violation Notice:  This is your"
        " final notice regarding the unpaid toll balance on your account. You must"
        " settle the balance within the next 12 hours to avoid severe penalties."
        "  Unpaid Balance: $6.68 Due Date: April 26, 2025  Failure to pay within"
        " this time frame will result in the following:  1.Immediate addition of"
        " late payment fees to your balance 2.Suspension of your vehicle"
        " registration by the Department of Motor Vehicles (DMV) 3.Collection"
        " actions, including a negative report to your credit file Please make"
        " your payment promptly to avoid these severe consequences and protect"
        " your driving privileges.  Pay Now:"
        "  https://e-zpassiag.com-etcsay.cc/us  If the link fails, reply with 'Y',"
        " exit the SMS, and reopen it to activate the link, or copy and paste it"
        " directly into your browser to complete your payment.  This is your last"
        " opportunity. Pay now to avoid irreversible consequences."
    )
    result = classify_message_ml(test_message)
    print(f"ML voting result: {result['label']}  (1: scam, 0: not scam)")
    print(f"Votes: {result['votes']}")
    print(f"Consensus status: {result['consensus_status']}")
    print(f"Reason: {result['reason']}")

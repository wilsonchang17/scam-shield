import joblib
from collections import Counter

vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
rf_model = joblib.load('models/random_forest_model.joblib')
dt_model = joblib.load('models/decision_tree_model.joblib')
xgb_model = joblib.load('models/xgboost_model.joblib')
knn_model = joblib.load('models/knn_model.joblib')


def classify_message_ml(message):
    print(f"ML votes: {message}")
    X = vectorizer.transform([message])

    preds = []
    preds.append(rf_model.predict(X)[0])
    preds.append(dt_model.predict(X)[0])
    preds.append(xgb_model.predict(X)[0])
    preds.append(knn_model.predict(X)[0])
    

    vote_count = Counter(preds)
    print(vote_count)
    #print(vote_count)
    if len(vote_count) == 1:
        return preds[0] 
    else:
        from llm_classifier import classify_message_llm
        llm_result = classify_message_llm(message)

    if llm_result == -1:
        return knn_model.predict(X)[0]
    
    return llm_result

if __name__ == '__main__':

    test_message = "I will like to have a meeting with professor about my project."
    result = classify_message_ml(test_message)
    print(f"ML voting result: {result}  (1: scam, 0: not scam)")



from fastapi import FastAPI
import pickle

app = FastAPI()

model = pickle.load(open("../saved_model/model.pkl", "rb"))
vectorizer = pickle.load(open("../saved_model/vectorizer.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "AI Feedback Analyzer API Running"}

@app.post("/predict")
def predict(feedback: str):
    vec = vectorizer.transform([feedback])
    prediction = model.predict(vec)[0]
    return {"sentiment": prediction}

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("../data/feedback.csv")

X = data["feedback"]
y = data["sentiment"]

# Convert text to vectors
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model + vectorizer
pickle.dump(model, open("../saved_model/model.pkl", "wb"))
pickle.dump(vectorizer, open("../saved_model/vectorizer.pkl", "wb"))

print("Model trained and saved!")

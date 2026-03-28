import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    "review": [
        "This product is amazing",
        "Worst product ever",
        "I love this item",
        "Fake and useless product",
        "Very good quality",
        "Do not buy this"
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = real, 0 = fake
}

df = pd.DataFrame(data)

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["review"])
y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Test prediction
test_review = ["This is a great product"]
test_vector = vectorizer.transform(test_review)
prediction = model.predict(test_vector)

print("Prediction:", "Real" if prediction[0] == 1 else "Fake")
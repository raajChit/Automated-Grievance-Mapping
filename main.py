import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset from CSV file
df = pd.read_csv('./synthetic_data.csv')

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df["grievance_text"], df["department"], test_size=0.2, random_state=42
)


# 3. Building TF-IDF + Random Forest Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
    ("clf", RandomForestClassifier())  # Using Random Forest for department classification
])
pipeline.fit(X_train, y_train)


# 4. Predicting Department and Evaluating Model
def evaluate_model():
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    
    # Print evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Call the evaluation function
evaluate_model()


import pandas as pd
import spacy
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)

data = pd.read_csv("data_with_chemistry.csv")
print("Categories found in dataset:", data['category'].unique())

data['abstract_clean'] = data['abstract'].astype(str).apply(preprocess)

X = data['abstract_clean']
y = data['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "svm": SVC(kernel='rbf', C=2.0, gamma='scale', probability=True),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    print(f"\nTraining {name} model...")
    pipeline = make_pipeline(TfidfVectorizer(), model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\nClassification report for {name}:")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, f"model_{name}.pkl")
    print(f"Saved model_{name}.pkl")

    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()
    print(f"Saved confusion_matrix_{name}.png")

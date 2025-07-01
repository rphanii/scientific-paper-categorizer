import gradio as gr
import joblib
import spacy

nlp = spacy.load("en_core_web_sm")
model = joblib.load("model_svm.pkl")

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)

def predict_category(abstract):
    if len(abstract.strip().split()) < 3:
        return "Please enter a longer abstract."
    clean_text = preprocess(abstract)
    prediction = model.predict([clean_text])[0]
    return prediction

gr.Interface(
    fn=predict_category,
    inputs=gr.Textbox(lines=6, placeholder="Enter abstract here..."),
    outputs="label",
    title="Scientific Paper Categorizer",
    description="Enter a scientific paper abstract to classify it into a category."
).launch()

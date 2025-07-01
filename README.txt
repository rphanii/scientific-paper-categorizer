
# Scientific Paper Categorizer

A machine learning application that classifies scientific paper abstracts into predefined categories such as Computer Science, Physics, Biology, Chemistry, and Mathematics.

## Features

- Classifies abstracts into 5 scientific domains
- Preprocessing with tokenization, stopword removal, and lemmatization
- Uses TF-IDF vectorization
- Trained with Logistic Regression, SVM, and Random Forest
- Evaluation with Accuracy, Precision, Recall, and F1-score
- Confusion matrix visualization
- Web app for real-time predictions


## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

- Source: arXiv dataset
- Format: CSV with columns `abstract` and `category`
- Contains 800 abstracts (200 per category)
- Added 200 abstracts(manually)

## Training

Run the training script to train and evaluate all models:

```bash
python train_model.py
```

Models will be saved as:

- `model_logistic.pkl`
- `model_svm.pkl`
- `model_random_forest.pkl`

Confusion matrices are saved as:

- `confusion_matrix_logistic.png`
- `confusion_matrix_svm.png`
- `confusion_matrix_random_forest.png`

## Web App

Launch the Gradio app:

```bash
python app.py
```

This will open a browser interface where you can input an abstract and get the predicted category.

## Files

- `data_with_chemistry.csv` - Dataset
- `train_model.py` - Training script
- `app.py` - Gradio web app
- `requirements.txt` - Python dependencies
- `README.md` - Project overview

## How to Use

1. Clone the repo
2. Create a virtual environment (optional)
3. Install requirements
4. Run `train_model.py` to train models
5. Run `app.py` to start the web app

## License

This project is for educational use only.

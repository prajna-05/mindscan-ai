# 🧬 MindScan AI — Mental Health Text Classifier

A machine learning powered web application that classifies 
text into 7 mental health categories using NLP techniques.

🔗 **Live Demo:** [Click here](your-streamlit-link-here)

---

## What it does

Input any text → Model predicts mental health category with confidence score

**7 Categories:** Normal | Depression | Anxiety | Suicidal | 
Stress | Bipolar | Personality Disorder

---

## Model Performance

| Model | Accuracy | AUC |
|---|---|---|
| **Logistic Regression** | **74.92%** | **0.921** |
| Linear SVM | 74.70% | 0.918 |
| Naive Bayes | 68.92% | 0.890 |
| Random Forest | 65.43% | 0.872 |

---

## Tech Stack

- **Language:** Python 3.10
- **NLP:** NLTK, TF-IDF Vectorization (8000 features)
- **ML:** scikit-learn (Logistic Regression)
- **Sentiment:** VADER
- **Explainability:** LIME
- **Frontend:** Streamlit
- **Dataset:** 53,043 Reddit posts (Kaggle)

---

## Features

- Single text prediction with confidence chart
- LIME explainability — see which words drove the prediction
- Batch prediction via CSV upload
- Crisis alert system for high-risk predictions
- Interactive probability distribution chart

---

## 📁 Project Structure

mental_health_nlp/
├── app_v2.py              ← Streamlit web app
├── requirements.txt       ← Dependencies
├── notebooks/
│   └── Mental_Health_NLP_Jupyter.ipynb
├── model/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
└── outputs/               ← EDA charts

---

## How to run locally

```bash
git clone https://github.com/yourusername/mindscan-ai
cd mindscan-ai
pip install -r requirements.txt
streamlit run app_v2.py
```

---

## Dataset

Kaggle — Sentiment Analysis for Mental Health  
53,043 Reddit posts across 7 mental health categories

---

## Disclaimer

This app is for research purposes only. 
Not a substitute for professional mental health diagnosis.


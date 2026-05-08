# 🧬 MindScan AI — Mental Health Text Classifier

NLP-powered web app that classifies text into 7 mental health categories using Machine Learning.

🔗 **Live Demo:** [Click here](your-streamlit-link-here)

---

##  What it does

Input any text and the model predicts the mental health category with confidence score.

**7 Categories:** Normal | Depression | Anxiety | Suicidal | Stress | Bipolar | Personality Disorder

---

##  Model Performance

| Model | Accuracy | AUC |
|---|---|---|
| **Logistic Regression** | **74.92%** | **0.921** |
| Linear SVM | 74.70% | 0.918 |
| Naive Bayes | 68.92% | 0.890 |
| Random Forest | 65.43% | 0.872 |

---

##  Features

- Single text prediction with confidence chart
- LIME explainability — see which words drove the prediction
- Batch prediction via CSV upload
- Crisis alert for high-risk predictions
- Interactive probability distribution chart

---

##  Tech Stack

- **Language:** Python 3.10
- **NLP:** NLTK, TF-IDF Vectorization (8,000 features)
- **ML:** scikit-learn — Logistic Regression
- **Sentiment:** VADER
- **Explainability:** LIME
- **Frontend:** Streamlit
- **Dataset:** 53,043 Reddit posts (Kaggle)

---

##  Project Structure

    mental_health_nlp/
    ├── app_v2.py
    ├── requirements.txt
    ├── notebooks/
    │   └── Mental_Health_NLP_Jupyter.ipynb
    ├── model/
    │   ├── best_model.pkl
    │   └── tfidf_vectorizer.pkl
    ├── outputs/
    │   └── (EDA charts)
    └── tableau_export/
        └── (dashboard CSVs)

---

##  Run Locally

    git clone https://github.com/prajna-05/mindscan-ai
    cd mindscan-ai
    pip install -r requirements.txt
    streamlit run app_v2.py

---

##  Dataset

Kaggle — Sentiment Analysis for Mental Health
53,043 Reddit posts across 7 mental health categories

---

##  Disclaimer

This app is for research purposes only.
Not a substitute for professional mental health diagnosis.
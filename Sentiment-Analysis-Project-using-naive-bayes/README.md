# 📊 Product Review Sentiment Analysis using Naive Bayes

This is a simple sentiment analysis project built from scratch using only **NumPy** and **Pandas**, with a **Streamlit** app interface for interactive predictions.

## 🔍 Features

- Custom Naive Bayes classifier for positive/negative sentiment
- Real-time prediction with confidence score
- Streamlit app with user-friendly interface
- Clean, modular Python code

## 📁 Project Structure

```
SentimentAnalysisNaiveBayes/
├── app/
│   ├── app.py         ← Streamlit UI
│   └── senti.py       ← Naive Bayes implementation
├── data/
│   └── product_reviews.csv  ← Product reviews dataset
├── requirements/
│   └── requirements.txt     ← Required packages
├── notebooks/
│   └── demo_predictions.ipynb  ← Sample predictions
├── reports/
└── README.md
```

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements/requirements.txt
```

2. Run the Streamlit app:
```bash
cd app
streamlit run app.py
```

## ✨ Example

Input:
> "This product works perfectly and exceeded my expectations."

Output:
> ✅ Sentiment: POSITIVE — Confidence: 92.7%

---

## 📦 Requirements

- pandas
- numpy
- streamlit

---

## 📬 Author

Made with ❤️ using only basic Python libraries.

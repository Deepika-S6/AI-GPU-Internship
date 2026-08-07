# InternGuard: AI-Based Fake Internship Offer Detection System

InternGuard is a one-day AI/ML capstone project that detects whether an internship offer looks genuine or suspicious.

## Problem Statement

Students often receive internship messages through WhatsApp, Telegram, email, and social media. Some posts ask for registration fees, promise guaranteed jobs, or provide vague work details. InternGuard helps students quickly screen such offers using machine learning and simple warning-signal rules.

## Features

- Classifies internship text as `genuine` or `suspicious`
- Accepts pasted text, uploaded PDF files, uploaded images, or public Google Forms links
- Shows confidence score
- Highlights warning signals such as payment requests, urgency, unrealistic promises, and vague roles
- Includes sample dataset
- Streamlit web interface

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- pypdf
- pytesseract
- requests
- TF-IDF Vectorizer
- Logistic Regression

## How To Run

```bash
cd "C:\Users\DEEPIKA S\OneDrive\Documents\New project\InternGuard"
pip install -r requirements.txt
streamlit run app.py
```

## Image OCR Note

Image upload uses Tesseract OCR. Install the Python package from `requirements.txt`, and install the Tesseract app on your system if image extraction shows an OCR error.

## Google Forms Note

Google Forms analysis works with public form links that can be opened without signing in. Private forms should be copied and pasted manually.

## Model Workflow

1. Load labeled internship offer samples from `internship_offers.csv`.
2. Convert text into numerical features using TF-IDF.
3. Train Logistic Regression classifier.
4. Predict whether new offer text is genuine or suspicious.
5. Show rule-based warning signals for explainability.

## Suggested Demo Input

```text
Urgent hiring for data science internship. No interview required.
Pay registration fee of 799 today and get guaranteed certificate, stipend, and job offer.
Contact only on WhatsApp.
```

Expected result: suspicious internship offer.

## Future Enhancements

- Add larger real-world dataset
- Check company email domain automatically
- Add URL safety detection
- Generate a downloadable PDF report
- Support multiple languages

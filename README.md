# Online Shopping Purchase Prediction Using Machine Learning

A Streamlit application for the UCI Online Shoppers Purchasing Intention Dataset.

## Team
- Sreehari J
- Rhea Sibi
- Richin
- Diya
- Chotu

## Features
- Business Problem page
- Data Insights page (upload a cleaned CSV to produce charts)
- Prediction page using the saved scikit-learn pipeline
- Purchase prediction and, where supported, purchase probability

## Local setup
Use Python 3.11 for a consistent environment.

```bash
python -m venv .venv
```

Activate:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows CMD: `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

Install and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal, usually http://localhost:8501.

## Google Colab testing
Upload/extract this project in Colab, then run:
```bash
!pip install -r requirements.txt
!streamlit run app.py & npx localtunnel --port 8501
```
Colab tunnels are temporary and are for testing only.

## Deployment
Push the project to GitHub. Deploy the repository using Streamlit Community Cloud and select `app.py` as the entry point. Keep the model in `models/`.

## Important
The pickle file was serialized with scikit-learn 1.6.1. The requirements file pins that version for compatibility. Only load pickle/joblib files from trusted sources.

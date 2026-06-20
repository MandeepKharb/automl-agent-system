# 🤖 AutoML Agent System

A **Multi-Agent Machine Learning Pipeline** built with LangGraph and Groq that autonomously analyses data, engineers features, trains multiple models, and generates a structured report — all from a single CSV upload.

> Built by [Mandeep Singh Kharb](https://www.linkedin.com/in/mandeep-singh-kharb-19b70928) · [YouTube: @DataScienceDiaries](https://youtube.com/@DataScienceDiaries) · [Book a 1:1 Session](https://topmate.io/mandeep_singh140)

---

## 🎥 Watch the Full Build Video

[![AutoML Agent System](https://img.shields.io/badge/YouTube-Watch%20Full%20Tutorial-red?style=for-the-badge&logo=youtube)](https://youtube.com/@DataScienceDiaries)

---

## ✨ What It Does

Upload any CSV dataset, type your problem statement, and watch 4 specialised AI agents work autonomously:

| Agent | Role | What It Does |
|-------|------|-------------|
| 🔍 Agent 1 | Data Analyst | Loads data · detects problem type · identifies patterns |
| ⚙️ Agent 2 | Feature Engineer | Fills missing values · encodes · TF-IDF for text |
| 🤖 Agent 3 | Model Trainer | Trains 6+ models · 5-fold CV · selects best |
| 📝 Agent 4 | Report Writer | Synthesises all findings into a structured report |

**Supports both Classification and Regression problems automatically.**

---

## 🏗️ Architecture

```
User Input (CSV + Problem Statement)
         │
         ▼
┌─────────────────────┐
│   Agent 1           │  ← LLM (Groq)
│   Data Analyst      │
└────────┬────────────┘
         │ LangGraph Shared State
         ▼
┌─────────────────────┐
│   Agent 2           │  ← LLM (Groq)
│   Feature Engineer  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Agent 3           │  ← scikit-learn (Local CPU)
│   Model Trainer     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Agent 4           │  ← LLM (Groq)
│   Report Writer     │
└────────┬────────────┘
         │
         ▼
Best Model + Accuracy + Full Report
```

> **Note:** The LLM never sees your raw data. It only reads summaries.
> All model training happens locally on your machine via scikit-learn.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph |
| LLM | Groq — llama-3.3-70b-versatile |
| ML Training | scikit-learn |
| Text Vectorization | TF-IDF (scikit-learn) |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
multiagent_ml/
├── agents/
│   ├── data_analyst.py        # Agent 1 — analyses dataset
│   ├── feature_engineer.py    # Agent 2 — preprocesses data
│   ├── model_trainer.py       # Agent 3 — trains & selects models
│   └── report_writer.py       # Agent 4 — generates report
├── tools/
│   ├── data_tools.py          # Dataset loading & analysis functions
│   ├── feature_tools.py       # Preprocessing & TF-IDF functions
│   └── model_tools.py         # Model training & evaluation functions
├── graph/
│   └── workflow.py            # LangGraph pipeline definition
├── state/
│   └── ml_state.py            # Shared state across all agents
├── ui/
│   └── app.py                 # Streamlit frontend
├── data/                      # Place your CSV datasets here
├── .env                       # API keys (not committed to git)
├── requirements.txt
└── main.py                    # CLI entry point
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MandeepKharb/multiagent_ml.git
cd multiagent_ml
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to **API Keys** → Create a new key
4. Copy the key

### 5. Create Your `.env` File

Create a file named `.env` in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 6. Run the Application

**Option A — Streamlit UI (recommended)**
```bash
streamlit run ui/app.py
```
Then open `http://localhost:8501` in your browser.

**Option B — Command Line**
```bash
python main.py
```

---

## 🚀 How to Use

1. **Upload** any CSV dataset using the file uploader
2. **Select** the target column you want to predict
3. **Type** your problem statement (e.g. *"Predict whether a passenger survived the Titanic disaster"*)
4. **Click** Run Pipeline
5. Watch all 4 agents execute in real time
6. View model comparison results and download the final report

---

## 📊 Supported Problem Types

The system **automatically detects** whether your problem is:

**Classification** — models trained: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM

**Regression** — models trained: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN, SVR

**Text Data** — automatically detected and vectorized using TF-IDF (300 features) before training

---

## 📋 Requirements

```
langgraph
langchain
langchain-groq
pandas
scikit-learn==1.3.2
numpy
streamlit
python-dotenv
fpdf2
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧪 Test Datasets

You can test with any of these freely available datasets:

| Dataset | Problem Type | Download |
|---------|-------------|----------|
| Titanic | Classification | [Kaggle](https://www.kaggle.com/datasets/titanic) |
| House Prices | Regression | [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| Fake News | Classification (Text) | [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |

Place downloaded CSVs in the `data/` folder.

---

## ⚠️ Common Issues

**Groq API rate limit error (413)**
> The prompt is too large. This can happen with very wide datasets. The system automatically handles this by sending only summaries to the LLM, not raw data.

**scikit-learn DLL error on Windows**
> Install a specific version: `pip install scikit-learn==1.3.2`

**ModuleNotFoundError for graph or agents**
> Make sure you run the app from the project root directory, not from inside the `ui/` folder.
> ```bash
> cd multiagent_ml
> streamlit run ui/app.py
> ```

**Text column being dropped**
> The system detects text columns automatically using average word count. If your text column is being dropped, check that it contains actual sentences (avg > 8 words), not just short labels.

---

## 🤝 Want to Learn How to Build This?

This project is part of my YouTube series on building real-world AI/ML systems.

- 🎥 **YouTube:** [youtube.com/@DataScienceDiaries](https://youtube.com/@DataScienceDiaries)
- 💼 **LinkedIn:** [Mandeep Singh Kharb](https://www.linkedin.com/in/mandeep-singh-kharb-19b70928)
- 📅 **Book a 1:1 Session:** [topmate.io/mandeep_singh140](https://topmate.io/mandeep_singh140)

If you're a working engineer or fresher looking to break into AI/ML roles — I offer personalised career guidance, resume reviews, and project mentorship. Check the link above.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

⭐ **If this project helped you, please give it a star on GitHub — it helps others find it.**

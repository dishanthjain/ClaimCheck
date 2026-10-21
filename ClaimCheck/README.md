# ClaimCheck

**AI-Powered Academic & Technical Claim Verification — local prototype**

ClaimCheck searches a small local collection of documents and shows passages semantically related to a user-entered claim.

## Important limitation

This first version is an **evidence retrieval prototype**, not a fully trained fact-checking classifier. It does not automatically prove whether a claim is true or false. It deliberately reports **“Evidence found — review required”** when relevant passages are found, and **“Insufficient evidence”** when the collection does not return a relevant passage. Similarity scores are relevance rankings, not probabilities of truth.

## Features

- Local document collection stored in `data/knowledge_base.json`
- Semantic search using sentence-transformers embeddings
- Passage ranking using cosine similarity
- Source names and optional source links
- Downloadable JSON investigation report
- Search history during the current app session
- No paid API key required

## Requirements

- Python 3.10 or newer recommended
- VS Code
- Internet access for installing packages and downloading the embedding model on first use
- A few GB of free disk space is recommended for Python packages and the model

## Run on macOS (zsh)

```bash
cd ClaimCheck
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`, and may open it in your browser.

## Run on Windows (PowerShell)

```powershell
cd ClaimCheck
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Add your own documents

Open `data/knowledge_base.json` and add objects like:

```json
{
  "id": "doc-002",
  "title": "Document title",
  "source": "Course notes or official documentation",
  "url": "https://example.com/document",
  "text": "Paste a short passage from a trusted, legally reusable source."
}
```

Keep the file valid JSON: use double quotes, commas between entries, and no trailing comma after the final entry. Prefer official documentation, course materials you are allowed to use, or other reputable sources.

## How it works

1. The user enters a claim.
2. The app splits local documents into passages.
3. A pretrained sentence-transformers model converts the claim and passages into embeddings.
4. Cosine similarity ranks the passages by semantic relevance.
5. The app shows the passages, source metadata, and a cautious assessment.

## Suggested next improvements

1. Add a labelled dataset of claim/evidence pairs.
2. Train or evaluate a support/contradiction/insufficient-evidence classifier.
3. Add tests and retrieval metrics such as Precision@k and Recall@k.
4. Improve source metadata and document ingestion.
5. Only add a verdict classifier once it has been evaluated on held-out examples.

## Project structure

```text
ClaimCheck/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── knowledge_base.json
```

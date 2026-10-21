# 🔎 ClaimCheck — AI-Powered Claim Verification

**An evidence-first AI/ML application that retrieves relevant passages from a local knowledge base to help users investigate factual claims.**

ClaimCheck is a Python-based application built with Streamlit, Sentence Transformers, and Scikit-learn. It allows users to enter a claim, search a collection of documents for semantically relevant evidence, inspect retrieved passages, and understand when the available information is insufficient to support a conclusion.

Rather than automatically declaring every claim true or false, ClaimCheck emphasizes evidence retrieval, source inspection, and responsible interpretation of AI-generated relevance scores.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [System Requirements](#-system-requirements)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [How to Use ClaimCheck](#-how-to-use-claimcheck)
- [Example Claims](#-example-claims)
- [Understanding the Results](#-understanding-the-results)
- [Knowledge Base](#-knowledge-base)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Learning Outcomes](#-learning-outcomes)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 📖 Overview

In the digital world, people frequently encounter factual statements, technical claims, and information that may be incomplete or misleading. Evaluating these claims requires identifying relevant evidence and understanding whether that evidence actually supports a conclusion.

ClaimCheck explores how natural language processing and semantic search can help with this process.

Users enter a statement into the application. The system converts the claim and passages from a local document collection into numerical vector representations using a pretrained Sentence Transformer model. It then calculates cosine similarity scores to rank passages according to their semantic relevance.

The application displays retrieved passages, relevance scores, and an assessment based on the available evidence.

The current implementation is a **local evidence-retrieval prototype**, not a fully automated fact-checking system. It does not independently establish the truth or falsity of a claim.

## 🎯 Problem Statement

Traditional keyword-based searches can struggle to identify relevant information when a query and a document express the same concept using different words.

For example, the statement:

> "Testing data helps evaluate how well a machine learning model generalizes."

may be semantically related to a passage discussing held-out datasets and model evaluation, even if the wording differs.

ClaimCheck uses semantic similarity to identify potentially relevant passages and make evidence review more accessible.

The main challenge is to retrieve useful information while avoiding unsupported conclusions when the knowledge base does not contain sufficient evidence.

## 🎯 Project Objectives

- Develop an interactive application for investigating factual and technical claims.
- Apply natural language processing to represent sentences and passages numerically.
- Implement semantic retrieval using pretrained sentence embeddings.
- Rank retrieved passages using cosine similarity.
- Present relevant passages from identifiable local documents.
- Indicate when the available evidence appears insufficient.
- Help users distinguish semantic relevance from factual truth.
- Establish a foundation for future evidence-based claim verification systems.

## ✨ Key Features

### 1. Claim Input

Users can enter natural-language statements they want to investigate.

### 2. Semantic Evidence Retrieval

ClaimCheck uses the `all-MiniLM-L6-v2` Sentence Transformer model to represent text as embeddings and retrieve passages based on semantic similarity.

### 3. Ranked Evidence

Retrieved passages are ranked according to their relevance to the submitted claim.

### 4. Source Identification

Each retrieved passage is associated with its source document from the local knowledge base.

### 5. Evidence Sufficiency Assessment

The current application can report that relevant evidence was found for review or that the available collection did not return sufficiently relevant evidence.

An insufficient-evidence result does not mean that a claim is false.

### 6. Relevance Scores

The interface displays relevance scores to help users understand how closely retrieved passages match the claim.

These scores measure semantic similarity; they are not probabilities that a claim is true.

### 7. Investigation History

The application maintains investigation history within the current Streamlit session.

### 8. Report Export

Users can download an investigation report using the application's report-download functionality.

### 9. Interactive Web Interface

Streamlit provides an accessible interface with claim input, configurable evidence retrieval, result summaries, and source passages.

### 10. Local Knowledge Base

The starter version retrieves evidence from a local JSON document collection without requiring a live web-search service.

---

## ⚙️ How It Works

ClaimCheck follows an evidence-retrieval pipeline.

1. **Claim submission:** The user enters a statement to investigate.
2. **Knowledge-base loading:** The application reads documents from the local JSON knowledge base.
3. **Passage preparation:** Documents are divided into smaller text passages for retrieval.
4. **Embedding generation:** The Sentence Transformer converts the claim and passages into numerical vector representations.
5. **Similarity calculation:** Cosine similarity measures the semantic relationship between the claim and each passage.
6. **Evidence ranking:** Passages are sorted by their relevance scores.
7. **Result presentation:** The interface displays the retrieved passages, source names, scores, and evidence assessment.
8. **Human review:** The user examines the retrieved information before drawing a conclusion.

### Simplified Workflow

```text
       User submits a claim
                 |
                 v
       Load local documents
                 |
                 v
       Split documents into passages
                 |
                 v
       Generate sentence embeddings
                 |
                 v
       Calculate cosine similarity
                 |
                 v
       Rank relevant passages
                 |
                 v
       Display evidence and sources
                 |
                 v
       Review evidence and limitations
```

### Semantic Similarity

The system uses cosine similarity to compare embedding vectors.

For vectors A and B:

```text
                 A · B
Similarity = ---------------
             ||A|| × ||B||
```

The score measures the similarity between the vector representations. It does not establish whether a statement is correct, whether a source is reliable, or whether the retrieved passage logically supports the claim.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Pretrained sentence embedding model |
| Scikit-learn | Cosine similarity and retrieval calculations |
| NumPy | Numerical operations |
| JSON | Local document storage |
| Git and GitHub | Version control and project hosting |

### Model Information

The project uses the pretrained Sentence Transformer model:

`sentence-transformers/all-MiniLM-L6-v2`

The model is used to generate text embeddings for semantic retrieval. The current version does not train a custom machine learning model or fine-tune the pretrained model on a claim-verification dataset.

---

## 🏗️ Project Architecture

ClaimCheck follows a lightweight architecture designed for local execution.

### Presentation Layer

Streamlit provides the user interface for entering claims, selecting the number of evidence passages, viewing results, and exporting reports.

### Retrieval Layer

The Python application processes the claim, prepares document passages, generates embeddings, calculates similarity scores, and ranks candidate evidence.

### Data Layer

The JSON knowledge base stores the documents used by the application.

### Model Layer

The pretrained Sentence Transformer produces embeddings that enable semantic comparisons between the claim and the stored passages.

The current architecture does not include a live web-search integration, an external fact-checking API, or an independent natural-language inference model.

---

## 📁 Project Structure

```text
ClaimCheck/
│
├── app.py
│   └── Main Streamlit application and evidence-retrieval pipeline
│
├── requirements.txt
│   └── Python dependency declarations
│
├── README.md
│   └── Project documentation
│
└── data/
    └── knowledge_base.json
        └── Local documents used for evidence retrieval
```

The exact files may vary as the project evolves.

---

## 💻 System Requirements

Before running ClaimCheck, ensure that you have:

- Python 3.12 recommended for the starter environment.
- pip, Python's package installer.
- Visual Studio Code or another Python-compatible editor.
- An internet connection for initial dependency installation and model download.
- Sufficient disk space and memory for Python dependencies and the embedding model.

After the model has been downloaded and dependencies are installed, local investigations can run without live web access, provided the model files are available locally.

---

## 🚀 Installation and Setup

### Step 1: Clone the Repository

Replace the example repository URL with your actual GitHub repository URL.

```bash
git clone https://github.com/YOUR-USERNAME/ClaimCheck.git
cd ClaimCheck
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### Step 2: Create a Virtual Environment

A virtual environment keeps project dependencies separate from other Python installations.

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

If the `python3` command on macOS points to an unsuitable Python installation, use the path to your working Python 3.12 interpreter to create the environment.

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Verify the Project Files

Confirm that `app.py`, `requirements.txt`, and `data/knowledge_base.json` exist in the project directory.

---

## ▶️ Running the Application

With the virtual environment activated, run:

```bash
python -m streamlit run app.py
```

Streamlit will start a local development server. Open the local URL shown in your terminal, typically:

```text
http://localhost:8501
```

Keep the terminal running while using the application.

To stop the server, press `Ctrl + C` in the terminal.

---

## 🧪 How to Use ClaimCheck

1. Start the application.
2. Open the local Streamlit URL in your browser.
3. Enter a factual or technical statement in the claim input field.
4. Choose the number of evidence passages to retrieve.
5. Click **Find evidence**.
6. Review the assessment and relevance score.
7. Read the retrieved passages and inspect their source documents.
8. Decide whether the evidence is relevant and sufficient for further investigation.
9. Download the report if you need to save the investigation results.

For the best results, begin with claims related to the topics already represented in the local knowledge base.

---

## 📝 Example Claims

The following claims are useful for testing the current starter knowledge base.

### Programming

```text
C++ is generally faster than Python for computationally intensive tasks.
```

### Machine Learning

```text
Machine learning systems learn patterns from examples.
```

```text
Training data is used to fit a machine learning model.
```

```text
Accuracy measures the fraction of predictions that are correct.
```

```text
Held-out test data is used to evaluate how well a model generalizes.
```

### Testing Insufficient Evidence

```text
Every machine learning model is 100% accurate.
```

```text
Lamborghini is better than Ferrari.
```

The last example is outside the current knowledge base and is subjective without a defined comparison criterion. The application may return insufficient evidence because it has no relevant car-performance documents.

The retrieved passages and assessments depend on the actual contents of the knowledge base and the retrieval threshold configured in the application.

---

## 📊 Understanding the Results

### Evidence Found — Review Required

The system retrieved passages that meet its relevance criteria. These passages should be inspected to determine whether they genuinely support or contradict the claim.

This result is not a guarantee that the claim is true.

### Insufficient Evidence

The system did not retrieve sufficiently relevant evidence from the available collection.

Possible reasons include:

- The knowledge base does not cover the topic.
- The documents use substantially different terminology.
- The relevant information is missing from the stored passages.
- The retrieval threshold is too restrictive.
- The claim requires additional context.

An insufficient-evidence result is not equivalent to a false claim.

### Relevance Score

A higher semantic similarity score generally indicates that a passage is more closely related to the claim in embedding space.

However, semantic similarity alone cannot determine:

- Whether the claim is factually correct.
- Whether the source is trustworthy.
- Whether the passage supports or contradicts the claim.
- Whether the evidence is current.
- Whether important context is missing.

---

## 📚 Knowledge Base

ClaimCheck currently uses a small starter collection stored in:

`data/knowledge_base.json`

The initial documents cover subjects such as:

- Python and C++ performance.
- Semantic search and sentence embeddings.
- Machine learning evaluation.
- Situations involving insufficient evidence.

### Expanding the Knowledge Base

To investigate a broader range of claims, add more reliable documents to the JSON collection using the existing document structure.

When adding content:

- Prefer authoritative and verifiable sources.
- Include source titles and source URLs when available.
- Keep passages clear and focused.
- Include publication dates for time-sensitive information.
- Avoid treating unsupported statements as established facts.
- Test retrieval with claims that are both related and unrelated to each document.

The application's supported document format should be preserved when adding new entries.

---

## ⚠️ Limitations

ClaimCheck is an educational prototype, and its current capabilities have important limitations.

**Limited knowledge coverage:** The system can retrieve evidence only from the local documents provided to it.

**No live web search:** The current version does not automatically search the internet for new information.

**No independent truth verification:** Semantic similarity does not establish whether a claim is true or false.

**No custom-trained classifier:** The current model is a pretrained sentence embedding model used for retrieval, not a classifier trained specifically to verify claims.

**Source reliability is not automatically established:** The system does not independently validate every source's authority, accuracy, or credibility.

**Potential retrieval errors:** Semantically related passages may not directly support a claim. Relevant passages can also be missed.

**Subjective claims require context:** Statements such as "this car is better" need measurable criteria before they can be evaluated meaningfully.

**Time-sensitive claims require current sources:** A static local knowledge base may become outdated.

For these reasons, ClaimCheck should be used as an evidence-discovery aid rather than as a definitive fact-checking authority.

---

## 🔮 Future Enhancements

Potential improvements include:

- **Live web search:** Retrieve relevant evidence from current online sources.
- **Source credibility assessment:** Prioritize authoritative sources and identify publication dates.
- **Claim classification:** Explore natural-language inference models to distinguish support, contradiction, and neutral evidence.
- **Evidence aggregation:** Compare multiple sources before presenting an assessment.
- **Improved document processing:** Support PDF, TXT, and other document formats.
- **Larger knowledge base:** Add curated documents across science, technology, education, and other subjects.
- **Advanced retrieval:** Experiment with hybrid keyword and semantic search.
- **Improved explainability:** Highlight the exact text that supports or challenges a claim.
- **Evaluation dataset:** Measure retrieval precision, recall, and ranking quality using labeled examples.
- **Persistent investigation history:** Store investigations across sessions.
- **Automated testing:** Add unit tests for document processing, ranking, and result handling.
- **Deployment:** Explore hosting the application for remote demonstrations.

These are potential future improvements, not claims about features already implemented.

---

## 🎓 Learning Outcomes

This project provides practical experience with:

- Natural language processing.
- Sentence embeddings and vector representations.
- Semantic search and information retrieval.
- Cosine similarity.
- Pretrained transformer models.
- Python application development.
- Streamlit interface design.
- JSON-based data management.
- Responsible interpretation of machine learning outputs.
- Git and GitHub project management.

It also demonstrates an important AI engineering principle: a useful system should communicate the limitations of its evidence rather than make unsupported claims.

---

## 🤝 Contributing

Contributions and suggestions are welcome.

To propose an improvement:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test the application.
5. Submit a pull request describing the improvement.

Please ensure that additional documents are appropriately sourced and that new features do not present semantic similarity scores as probabilities of factual truth.


---

## 👨‍💻 Author

**Dishanth Jain**

Computer Science Engineering Student

GitHub: [@dishanthjain](https://github.com/dishanthjain)

---

## ⭐ Support

If you find this project useful for learning about NLP, semantic search, and evidence-based claim investigation, consider starring the repository on GitHub.

**ClaimCheck — Retrieve evidence. Examine sources. Avoid unsupported conclusions.**

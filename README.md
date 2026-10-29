# Information-Ritrival

# Comparative Study of Text Classification Algorithms

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Topic](https://img.shields.io/badge/Domain-Information%20Retrieval%20%7C%20NLP-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

A complete, self-contained Information Retrieval and Text Mining benchmark suite implemented in pure **Python**, **NumPy**, and **Pandas**. 

This repository evaluates four classical machine learning classifiers built **from scratch** (without using `scikit-learn` model implementations) on a multi-class document classification task using a custom-engineered TF-IDF feature extraction pipeline.

---

## 📁 Repository Structure

```text
Information Retrieval/
│
├── dataset.csv                 # Benchmark dataset (1,750 labeled text documents)
├── run_experiment.py           # End-to-end pipeline (Data gen, TF-IDF, Models, Evaluation)
├── comparative_results.csv     # Model evaluation metrics (Accuracy, Precision, Recall, F1)
├── what_if_results.csv         # Sensitivity analysis varying KNN neighbors (k)
└── README.md                   # Project documentation and guide
```

### Detailed File Overview

| File | Type | Description |
| :--- | :--- | :--- |
| [`run_experiment.py`](run_experiment.py) | Python Script | Main execution pipeline. Handles dataset generation/loading, tokenization, vocabulary building (top 2,500 terms), custom TF-IDF calculation, model definitions, evaluation, and exporting results. |
| [`dataset.csv`](dataset.csv) | CSV Dataset | Labeled corpus containing 1,750 text documents across 5 categories (`tech`, `business`, `sport`, `entertainment`, `politics`), with 350 samples per class. |
| [`comparative_results.csv`](comparative_results.csv) | CSV Output | Benchmark output table comparing classification metrics (Accuracy, Precision, Recall, Macro F1) across all four classifiers. |
| [`what_if_results.csv`](what_if_results.csv) | CSV Output | Experimental results recording Macro F1-score across various neighborhood sizes ($k \in \{1, 3, 5, 7, 11, 15\}$) for the K-Nearest Neighbors classifier. |

---

## 🚀 Key Features

- **Built from First Principles**:
  - Feature extraction and classification algorithms are written with raw NumPy linear algebra and vector operations.
  - Zero dependencies on external ML frameworks such as `scikit-learn`, `nltk`, or `spacy`.
- **Custom NLP & Feature Engineering Pipeline**:
  - Regex-based text cleaning and tokenization.
  - Domain and English stopword filtering.
  - Top-2,500 vocabulary indexing.
  - Sublinear TF-IDF calculation with log-scaled term frequencies, smoothed inverse document frequency ($IDF$), and $L_2$ document normalization.
- **Four Distinct Classifier Architectures**:
  1. **Multinomial Naïve Bayes** with Laplace smoothing ($\alpha = 0.1$).
  2. **Linear Support Vector Machine (Linear SVM)** trained with subgradient descent optimizing Hinge Loss.
  3. **K-Nearest Neighbors ($k$-NN)** using Cosine distance / normalized dot-product similarities.
  4. **Decision Tree (CART-style)** using Gini Impurity split criteria (max depth = 8).
- **Automated Metric Suite**:
  - Confusion matrix derivation, Accuracy, Macro Precision, Macro Recall, and Macro F1-score.
- **Hyperparameter Sensitivity / "What-If" Analysis**:
  - Automated experiment analyzing performance variation across values of $k$ in KNN.

---

## 📊 Dataset Specifications

The dataset (`dataset.csv`) contains structured rows with two fields:
- `text`: Synthetic document generated from domain-specific lexicons with natural English filler words.
- `category`: Ground truth topic category.

### Category Breakdown (1,750 total documents)
- `tech`: 350 documents (vocabulary includes *software, computer, ai, cyber, algorithm, server*)
- `business`: 350 documents (vocabulary includes *market, economy, shares, bank, inflation, revenue*)
- `sport`: 350 documents (vocabulary includes *match, goal, team, league, tournament, stadium*)
- `entertainment`: 350 documents (vocabulary includes *film, movie, music, concert, award, theater*)
- `politics`: 350 documents (vocabulary includes *government, election, minister, party, parliament, policy*)

---

## 🛠️ Installation & Requirements

The project requires Python 3.8+ and standard scientific packages.

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy pandas
```

---

## ⚡ How to Run

Run the end-to-end experiment script directly:

```bash
python run_experiment.py
```

### Script Execution Flow
1. **Checks for Dataset**: Loads existing `dataset.csv` (or generates it automatically if missing).
2. **Preprocessing**: Cleans, tokenizes, and extracts TF-IDF representations for training (80%) and testing (20%) sets.
3. **Training & Inference**: Trains each classifier and predicts class labels on the test set.
4. **Evaluation**: Prints the comparative metric table and exports to `comparative_results.csv`.
5. **What-If Experiment**: Runs $k$-NN over $k \in [1, 3, 5, 7, 11, 15]$ and exports results to `what_if_results.csv`.

---

## 📈 Experimental Results

### 1. Comparative Classifier Benchmark

*Saved to [`comparative_results.csv`](comparative_results.csv)*

| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **KNN ($k=7$)** | 100.00 | 20.00 | 20.00 | 20.00 |
| **Naïve Bayes ($\alpha=0.1$)** | 100.00 | 20.00 | 20.00 | 20.00 |
| **Decision Tree ($d_{max}=8$)** | 0.00 | 0.00 | 0.00 | 0.00 |
| **Linear SVM ($C=1.0$)** | 0.00 | 0.00 | 0.00 | 0.00 |

### 2. What-If Experiment: Impact of Neighborhood Size ($k$) on KNN

*Saved to [`what_if_results.csv`](what_if_results.csv)*

| $k$ Neighbours | Macro F1 (%) |
| :---: | :---: |
| 1 | 20.00 |
| 3 | 20.00 |
| 5 | 20.00 |
| 7 | 20.00 |
| 11 | 20.00 |
| 15 | 20.00 |

---

## 💡 Technical Note on Train-Test Partitioning

> [!TIP]
> In the default configuration of `run_experiment.py`, the dataset is generated category-by-category and partitioned sequentially:
> ```python
> train_size = int(0.8 * len(df))
> train_df = df.iloc[:train_size]
> test_df = df.iloc[train_size:]
> ```
> Because documents are ordered sequentially by category, the first 80% contains `tech`, `business`, `sport`, and `entertainment`, while the remaining 20% contains only `politics`.
>
> **Recommended Enhancement**:
> To obtain balanced, generalized classification performance across all 5 classes, shuffle the dataset before splitting:
> ```python
> df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
> ```

---

## 📜 Mathematical Overview of Models

- **TF-IDF Calculation**:
  $$\text{TF}(t, d) = 1 + \log(f_{t, d}) \quad \text{for } f_{t, d} > 0$$
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$$
- **Multinomial Naïve Bayes**:
  $$P(c \mid d) \propto P(c) \prod_{t \in d} P(t \mid c)$$
  with Laplace smoothing:
  $$P(t \mid c) = \frac{\sum f_{t, c} + \alpha}{\sum_{t'} \sum f_{t', c} + \alpha \cdot |V|}$$
- **Linear SVM (One-vs-Rest)**:
  Minimizes empirical hinge loss with regularized subgradient updates:
  $$L(w, b) = \frac{1}{2n} \|w\|^2 + C \cdot \max(0, 1 - y_i(w^T x_i + b))$$
- **$k$-Nearest Neighbors**:
  Computes distance via normalized dot-product (cosine similarity) and predicts via majority voting:
  $$\text{dist}(x, z) = 1 - x \cdot z^T$$
- **Decision Tree**:
  Recursive binary feature splitting evaluating split gain via Gini Impurity:
  $$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).


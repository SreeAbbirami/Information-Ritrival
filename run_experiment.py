import os
import re
import numpy as np
import pandas as pd

print("=======================================================================")
print(" TEXT CLASSIFICATION COMPARATIVE STUDY (LOCAL DATASET GENERATOR)       ")
print("=======================================================================")

# ---------------------------------------------------------
# 1. AUTO-CREATE OR LOAD LOCAL DATASET (dataset.csv)
# ---------------------------------------------------------
DATASET_FILENAME = "dataset.csv"

if not os.path.exists(DATASET_FILENAME):
    print(f"[INFO] '{DATASET_FILENAME}' not found. Generating benchmark text dataset...")
    np.random.seed(42)
    categories = ['tech', 'business', 'sport', 'entertainment', 'politics']
    
    vocab = {
        'tech': ['software', 'computer', 'apple', 'microsoft', 'iphone', 'internet', 'data', 'cyber', 'code', 'algorithm', 'startup', 'digital', 'server', 'ai'],
        'business': ['market', 'economy', 'shares', 'bank', 'inflation', 'investors', 'company', 'profit', 'revenue', 'trade', 'stock', 'financial', 'ceo', 'growth'],
        'sport': ['match', 'goal', 'team', 'player', 'league', 'cup', 'coach', 'tournament', 'score', 'win', 'game', 'club', 'stadium', 'race', 'season'],
        'entertainment': ['film', 'movie', 'actor', 'director', 'music', 'album', 'concert', 'star', 'award', 'show', 'theater', 'broadcast', 'song', 'celebrity'],
        'politics': ['government', 'election', 'minister', 'party', 'vote', 'parliament', 'policy', 'leader', 'campaign', 'law', 'state', 'public', 'debate', 'summit']
    }
    
    data_rows = []
    for cat in categories:
        for _ in range(350): # 350 samples per category = 1,750 total rows
            words = list(np.random.choice(vocab[cat], size=np.random.randint(20, 40), replace=True))
            # add some random English filler words
            filler = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'for', 'it', 'with', 'as', 'was', 'on']
            words += list(np.random.choice(filler, size=15, replace=True))
            np.random.shuffle(words)
            data_rows.append({"text": " ".join(words), "category": cat})
            
    df_gen = pd.DataFrame(data_rows)
    df_gen.to_csv(DATASET_FILENAME, index=False)
    print(f"[INFO] Successfully created '{DATASET_FILENAME}' with {len(df_gen)} labelled documents!")

# Load the dataset
df = pd.read_csv(DATASET_FILENAME)
CATEGORIES = df['category'].unique().tolist()
print(f"[INFO] Loaded Dataset from '{DATASET_FILENAME}' | Classes: {CATEGORIES}")

# Train/Test Split
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

raw_train_docs = train_df['text'].values
y_train = pd.factorize(train_df['category'])[0]
raw_test_docs = test_df['text'].values
y_test = pd.factorize(test_df['category'])[0]

print(f"[INFO] Training Samples : {len(raw_train_docs)}")
print(f"[INFO] Testing Samples  : {len(raw_test_docs)}")

# ---------------------------------------------------------
# 2. TEXT PREPROCESSING & TF-IDF NUMERICAL FEATURE EXTRACTION
# ---------------------------------------------------------
stopwords = {"the", "and", "to", "of", "a", "in", "is", "that", "for", "it", "with", "as", "was", "on"}

def clean_text(text: str) -> list:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return [w for w in text.split() if len(w) >= 3 and w not in stopwords]

train_tokens = [clean_text(doc) for doc in raw_train_docs]
test_tokens = [clean_text(doc) for doc in raw_test_docs]

# Build Vocabulary (Top 2500 features)
all_words = [w for doc in train_tokens for w in doc]
vocab_counts = pd.Series(all_words).value_counts().head(2500).index.tolist()
word_to_idx = {w: idx for idx, w in enumerate(vocab_counts)}
V = len(vocab_counts)
N_train = len(train_tokens)

# Compute IDF
df_counts = np.zeros(V, dtype=np.float64)
for doc in train_tokens:
    for w in set(doc):
        if w in word_to_idx:
            df_counts[word_to_idx[w]] += 1.0

idf = np.log((1.0 + N_train) / (1.0 + df_counts)) + 1.0

def compute_tfidf(tokenized_docs, l2_norm=True):
    mat = np.zeros((len(tokenized_docs), V), dtype=np.float64)
    for i, doc in enumerate(tokenized_docs):
        for w in doc:
            if w in word_to_idx:
                mat[i, word_to_idx[w]] += 1.0
        mask = mat[i] > 0
        mat[i, mask] = (1.0 + np.log(mat[i, mask])) * idf[mask]
        if l2_norm:
            norm = np.linalg.norm(mat[i])
            if norm > 0:
                mat[i] /= norm
    return mat

X_train = compute_tfidf(train_tokens, l2_norm=True)
X_test = compute_tfidf(test_tokens, l2_norm=True)
print(f"[INFO] TF-IDF Feature Matrix Constructed: Train={X_train.shape}, Test={X_test.shape}")

# ---------------------------------------------------------
# 3. IMPLEMENT THE 4 CLASSIFIERS
# ---------------------------------------------------------
class NaiveBayes:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = np.zeros(len(self.classes))
        self.likelihood = np.zeros((len(self.classes), X.shape[1]))
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.prior[idx] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.likelihood[idx] = np.log(counts / counts.sum())
        return self
    def predict(self, X):
        jll = X @ self.likelihood.T + self.prior
        return self.classes[np.argmax(jll, axis=1)]

class LinearSVM:
    def __init__(self, C=1.0, lr=0.1, epochs=150):
        self.C, self.lr, self.epochs = C, lr, epochs
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.W = np.zeros((len(self.classes), n_features))
        self.b = np.zeros(len(self.classes))
        for idx, c in enumerate(self.classes):
            y_bin = np.where(y == c, 1.0, -1.0)
            w = np.zeros(n_features)
            b = 0.0
            for _ in range(self.epochs):
                margins = y_bin * (X @ w + b)
                violated = margins < 1.0
                grad_w = w / n_samples - self.C * ((y_bin[violated, None] * X[violated]).mean(axis=0) if np.any(violated) else 0.0)
                grad_b = -self.C * (y_bin[violated].mean() if np.any(violated) else 0.0)
                w -= self.lr * grad_w
                b -= self.lr * grad_b
            self.W[idx], self.b[idx] = w, b
        return self
    def predict(self, X):
        scores = X @ self.W.T + self.b
        return self.classes[np.argmax(scores, axis=1)]

class KNN:
    def __init__(self, k=7):
        self.k = k
    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        return self
    def predict(self, X):
        sim = X @ self.X_train.T
        dists = 1.0 - sim
        knn_indices = np.argsort(dists, axis=1)[:, :self.k]
        preds = [np.bincount(self.y_train[row]).argmax() for row in knn_indices]
        return np.array(preds)

class DecisionTree:
    def __init__(self, max_depth=8):
        self.max_depth = max_depth
    def _gini(self, y):
        if len(y) == 0: return 0.0
        probs = np.bincount(y, minlength=len(CATEGORIES)) / len(y)
        return 1.0 - np.sum(probs ** 2)
    def _build(self, X, y, depth):
        n_samples, n_feat = X.shape
        if depth >= self.max_depth or len(np.unique(y)) == 1 or n_samples < 5:
            return {'leaf': True, 'class': int(np.bincount(y, minlength=len(CATEGORIES)).argmax())}
        best_feat, best_thresh, best_gain = None, None, -1.0
        parent_g = self._gini(y)
        for f in range(min(40, n_feat)):
            col = X[:, f]
            nz = col[col > 0]
            if len(nz) == 0: continue
            thresh = float(np.median(nz))
            left = col <= thresh
            if np.sum(left) == 0 or np.sum(~left) == 0: continue
            gain = parent_g - (np.sum(left)/n_samples)*self._gini(y[left]) - (np.sum(~left)/n_samples)*self._gini(y[~left])
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, f, thresh
        if best_gain <= 1e-5:
            return {'leaf': True, 'class': int(np.bincount(y, minlength=len(CATEGORIES)).argmax())}
        left = X[:, best_feat] <= best_thresh
        return {'leaf': False, 'feat': best_feat, 'thresh': best_thresh,
                'left': self._build(X[left], y[left], depth+1),
                'right': self._build(X[~left], y[~left], depth+1)}
    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self
    def _pred_one(self, x, node):
        if node['leaf']: return node['class']
        if x[node['feat']] <= node['thresh']: return self._pred_one(x, node['left'])
        return self._pred_one(x, node['right'])
    def predict(self, X):
        return np.array([self._pred_one(x, self.tree) for x in X])

# ---------------------------------------------------------
# 4. EVALUATION PROTOCOL (Accuracy, Precision, Recall, F1)
# ---------------------------------------------------------
def evaluate(y_true, y_pred):
    n_classes = len(CATEGORIES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    acc = np.trace(cm) / np.sum(cm)
    precs, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        precs.append(p); recalls.append(r); f1s.append(f)
    return acc*100, np.mean(precs)*100, np.mean(recalls)*100, np.mean(f1s)*100, cm

models = {
    "Decision Tree": DecisionTree(max_depth=8),
    "KNN (k=7)": KNN(k=7),
    "Linear SVM": LinearSVM(C=1.0),
    "Naïve Bayes": NaiveBayes(alpha=0.1)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc, prec, rec, f1, cm = evaluate(y_test, preds)
    results.append({
        "Classifier": name,
        "Accuracy (%)": round(acc, 2),
        "Precision (%)": round(prec, 2),
        "Recall (%)": round(recall := rec, 2),
        "F1-Score (%)": round(f1, 2)
    })

results_df = pd.DataFrame(results)
print("\n=================================================================")
print("               COMPARATIVE PERFORMANCE TABLE                     ")
print("=================================================================")
print(results_df.to_string(index=False))
results_df.to_csv("comparative_results.csv", index=False)

# ---------------------------------------------------------
# 5. WHAT-IF EXPERIMENT: VARYING KNN NEIGHBOURS (k)
# ---------------------------------------------------------
what_if_rows = []
for k_val in [1, 3, 5, 7, 11, 15]:
    knn_exp = KNN(k=k_val).fit(X_train, y_train)
    _, _, _, f1_val, _ = evaluate(y_test, knn_exp.predict(X_test))
    what_if_rows.append({"k_neighbours": k_val, "Macro_F1 (%)": round(f1_val, 2)})

what_if_df = pd.DataFrame(what_if_rows)
print("\n=================================================================")
print("   WHAT-IF EXPERIMENT RESULTS (KNN k-value variation)          ")
print("=================================================================")
print(what_if_df.to_string(index=False))
what_if_df.to_csv("what_if_results.csv", index=False)
print("\n[SUCCESS] Files saved: 'dataset.csv', 'comparative_results.csv', 'what_if_results.csv'")
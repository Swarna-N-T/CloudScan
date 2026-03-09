# 🔍 RGCN Folder — Complete Analysis

## 📁 File Overview

| File | Purpose |
|---|---|
| `label_dataset.py` | Runs **Checkov** on Terraform repos → generates `checkov_report.json` per repo |
| `build_graphs.py` | Parses `.tf` files → builds **NetworkX** directed graphs with risk labels |
| `cloud_dataset.py` | Converts raw `.gpickle` graphs → **PyTorch Geometric** `Data` objects |
| `rgcn_model.py` | Defines the **RGCN** model architecture |
| `rgcn_train.py` | Training loop with train/val/test split and evaluation |
| `visualize.py` | Visualizes a single graph using `matplotlib` |

---

## 🧠 Graph Definition

| Aspect | Detail |
|---|---|
| **Nodes** | **AWS Terraform Resources** (e.g., `aws_s3_bucket`, `aws_iam_role`, `aws_security_group`). Each `.tf` resource block becomes a node. |
| **Edges** | **Dependencies + Permissions** — Two types of directed edges: `dependency` (type `0`) — one resource references another; `permission` (type `1`) — edge involves a security-related resource (IAM, policy, security group, ACL) |
| **Node Features** | The **node type index** (integer). Each unique AWS resource type is mapped to an integer via a vocabulary (`node_type_map`). This integer is embedded via `torch.nn.Embedding` in the model. **No continuous feature vectors** — just a single categorical feature per node. |
| **Number of Relations** | **2** — `dependency` (0) and `permission` (1) |

---

## 🏗️ Framework

| Question | Answer |
|---|---|
| **PyTorch Geometric?** | ✅ **Yes.** Uses `RGCNConv`, `Data`, `InMemoryDataset`, and `DataLoader` from `torch_geometric` |

---

## 🎯 Task

| Question | Answer |
|---|---|
| **Node classification OR Graph classification?** | **Node Classification.** Each node is classified into a risk score. Loss is computed per-node, accuracy is counted per node (`data.num_nodes`). |
| **Classes** | **4 classes**: `0` (Safe/No risk), `1` (Low), `2` (Medium), `3` (High) |

---

## 📊 Train/Test Split

| Split | Ratio | Method |
|---|---|---|
| **Train** | **80%** | `dataset[:train_idx]` |
| **Validation** | **10%** | `dataset[train_idx:val_idx]` |
| **Test** | **10%** | `dataset[val_idx:]` |
| **Shuffle** | ✅ `dataset.shuffle()` before splitting |
| **Cross-validation?** | ❌ **No.** Single random split only, no k-fold CV |

---

## 📈 Evaluation Metrics

| Metric | Used? | Details |
|---|---|---|
| **Accuracy** | ✅ Yes | `correct_nodes / total_nodes` for both val and test |
| **F1 Score** | ✅ Yes | `sklearn.metrics.f1_score` with `average='macro'` |
| **Precision/Recall** | ❌ No | Not separately reported |
| **Confusion Matrix** | ✅ Yes | Printed on test set via `sklearn.metrics.confusion_matrix` |

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| **Hidden Dimension** | `64` |
| **Epochs** | `50` |
| **Learning Rate** | `0.01` |
| **Batch Size** | `32` |
| **Optimizer** | Adam |
| **Loss Function** | `CrossEntropyLoss` with **inverse class-frequency weights** |
| **Dropout** | `0.2` (after each RGCN layer and in the MLP decoder) |
| **num_bases** | `30` (basis decomposition for RGCN weight matrices) |
| **Model checkpointing** | ✅ Best model by val accuracy → `rgcn_model.pth` |

---

## 📦 Dataset Characteristics

### Raw Dataset (Input)

| Property | Detail |
|---|---|
| **Source** | `AWSonly_graph_Dataset` — a collection of open-source Terraform repositories targeting AWS |
| **Format** | Each repository is a folder containing one or more `.tf` (Terraform HCL) files |
| **Max Repos Processed** | **1,500** (configured via `MAX_REPOS` in `build_graphs.py`) |
| **Labeling Tool** | **Checkov** — static analysis scanner for IaC, run per repo to generate `checkov_report.json` |

### Parsed Details (Graph Construction — `build_graphs.py`)

| Property | Detail |
|---|---|
| **Parser** | `python-hcl2` — converts `.tf` HCL files into Python dictionaries |
| **Node Extraction** | Each `resource` block in a `.tf` file becomes a **node** (e.g., `aws_s3_bucket`, `aws_iam_role`) |
| **Node Attributes** | `type` (normalized AWS resource type), `config` (raw resource configuration), `risk_score` (0–3) |
| **Risk Score Assignment** | Checkov failed checks → severity mapping: Explicit severity (`CRITICAL`/`HIGH` → 3, `MEDIUM` → 2, `LOW` → 1), Heuristic keyword matching, Default → 2. Max score kept per resource. Resources with no failed checks → 0 (Safe). |
| **Edge Extraction** | Two heuristics: (1) **Direct reference** — source config contains target's Checkov-style ID; (2) **Token overlap** — fuzzy matching via shared unique tokens (min length 4, stop-word filtered) |
| **Edge Types** | `dependency` (type 0) — general reference; `permission` (type 1) — involves IAM/security-related resources |
| **Filter Criteria** | Only graphs with **≥ 1 node AND ≥ 1 edge** are retained and serialized as `.gpickle` |

### Preprocessed Details (PyTorch Geometric Conversion — `cloud_dataset.py`)

| Property | Detail |
|---|---|
| **Vocabulary Construction** | Two-pass process: (1) Scan all `.gpickle` files to collect unique AWS resource types; (2) Build sorted integer mapping (`node_type_map`) |
| **Node Feature (`x`)** | Single integer index per node — the resource type's position in the global vocabulary. **No continuous features.** |
| **Node Label (`y`)** | `risk_score` attribute (0 = Safe, 1 = Low, 2 = Medium, 3 = High) |
| **Edge Index** | Node IDs remapped to contiguous `0…N-1`; stored as `[2, E]` tensor |
| **Edge Type** | Integer tensor: `0` (dependency) or `1` (permission) |
| **Output Format** | `InMemoryDataset` → `data.pt` (graph data + slices) + `node_type_map.pkl` (vocabulary) |
| **Filter** | Graphs with 0 nodes or 0 edges are **skipped** during conversion |

### Sample Pipeline Summary

```
AWSonly_graph_Dataset (up to 1,500 repos)
  │
  ├─ label_dataset.py ──→ Adds checkov_report.json per repo (parallel, 4 threads)
  │
  ├─ build_graphs.py ──→ Parses .tf files → NetworkX DiGraphs → .gpickle files
  │                       (only graphs with nodes > 0 AND edges > 0 are saved)
  │
  └─ cloud_dataset.py ─→ Converts .gpickle → PyG Data objects → data.pt
                          (vocabulary built, node IDs remapped, features encoded)
```

> [!NOTE]
> Exact sample counts (total graphs, total nodes, class distribution) are printed at runtime by `rgcn_train.py`:
> - `Dataset Size: {n}` — total number of graphs
> - `Class Counts: {class_counts}` — per-class node distribution `[safe, low, med, high]`
> - `Train: {train}, Val: {val}, Test: {test}` — split sizes

---

## 🧪 Model Training & Evaluation Setup

### Dataset Split

| Split | Ratio | Size | Method |
|---|---|---|---|
| **Train** | **80%** | `dataset[:train_idx]` | Used for gradient updates |
| **Validation** | **10%** | `dataset[train_idx:val_idx]` | Model selection (best checkpoint by accuracy) |
| **Test** | **10%** | `dataset[val_idx:]` | Final evaluation (loaded from best checkpoint) |
| **Shuffle** | — | `dataset.shuffle()` | Random permutation before splitting |

> [!IMPORTANT]
> The split is performed at the **graph level** (each graph = one Terraform repo). However, metrics (accuracy, F1, confusion matrix) are computed at the **node level** (total correct nodes / total nodes).

### Class Imbalance Handling

- Class frequencies are computed from **training set node labels only**
- **Inverse class-frequency weights**: `weight[c] = total_samples / (num_classes × count[c])`
- Weights are passed to `CrossEntropyLoss` to penalize misclassification of minority risk classes

### Training Configuration

| Component | Detail |
|---|---|
| **Optimizer** | Adam (`lr=0.01`) |
| **Loss Function** | `CrossEntropyLoss` with inverse-frequency class weights |
| **Epochs** | 50 |
| **Batch Size** | 32 (via PyG `DataLoader`) |
| **Checkpointing** | Best model by validation accuracy → `rgcn_model.pth` |
| **Cross-Validation** | ❌ No — single random 80/10/10 split |

### Training Loop (Per Epoch)

1. **Forward pass** — Batched graphs through RGCN; loss on **all nodes** in batch
2. **Backward pass** — Gradients via Adam
3. **Validation** — Accuracy (correct nodes / total nodes) + Macro F1
4. **Checkpoint** — Save if current val accuracy > best so far

---

## 📊 Performance Metrics — Confusion Matrix

### Metrics Reported

| Metric | Computation | Scope |
|---|---|---|
| **Accuracy** | `correct_nodes / total_nodes` | Validation (per epoch) + Test (final) |
| **Macro F1 Score** | `sklearn.metrics.f1_score(labels, preds, average='macro')` | Validation (per epoch) + Test (final) |
| **Confusion Matrix** | `sklearn.metrics.confusion_matrix(test_labels, test_preds)` | Test set only (final) |

### Confusion Matrix Structure

The confusion matrix is a **4 × 4** matrix corresponding to the 4 risk classes:

```
                  Predicted
                  Safe(0)  Low(1)  Med(2)  High(3)
Actual Safe(0)  [  TP₀      ...     ...     ...  ]
Actual Low(1)   [  ...      TP₁     ...     ...  ]
Actual Med(2)   [  ...      ...     TP₂     ...  ]
Actual High(3)  [  ...      ...     ...     TP₃   ]
```

| Cell | Meaning |
|---|---|
| **Diagonal (TPᵢ)** | Correctly classified nodes for class `i` |
| **Off-diagonal** | Misclassifications — row = actual class, column = predicted class |

### How to Interpret

- **High diagonal values** → model correctly identifies the risk class
- **Row sums** → total actual samples per class
- **Column sums** → total predicted samples per class
- **Off-diagonal patterns** → common misclassification pairs (e.g., Safe↔Low confusion indicates the model struggles with borderline cases)

### Evaluation Flow

```
Training Complete (50 epochs)
  │
  ├─ Load best checkpoint (rgcn_model.pth) based on val accuracy
  │
  ├─ Forward pass on held-out Test set (10% of graphs)
  │
  └─ Report:
       • Test Accuracy
       • Test Macro F1 Score
       • 4×4 Confusion Matrix (via sklearn)
```

> [!TIP]
> To get **per-class Precision and Recall**, add `classification_report` from sklearn:
> ```python
> from sklearn.metrics import classification_report
> print(classification_report(test_labels, test_preds,
>       target_names=['Safe', 'Low', 'Medium', 'High']))
> ```

---

## 🏛️ Model Architecture

```
Input: Node type index (integer per node)
  ↓
Embedding Layer (num_node_types → 64)
  ↓
RGCNConv Layer 1 (64 → 64, 2 relations, 30 bases) + ReLU + Dropout(0.2)
  ↓
RGCNConv Layer 2 (64 → 64, 2 relations, 30 bases) + ReLU + Dropout(0.2)
  ↓
Linear (64 → 64) + ReLU + Dropout(0.2)
  ↓
Linear (64 → 4)  ← logits for 4 risk classes
```

---

## 🛠️ Implementation Details

### 1. Experimental Setup — Software Specification

| Component | Specification |
|---|---|
| **Programming Language** | Python 3.x |
| **Deep Learning Framework** | PyTorch |
| **Graph Neural Network Library** | PyTorch Geometric (`torch_geometric`) — provides `RGCNConv`, `Data`, `InMemoryDataset`, `DataLoader` |
| **Graph Construction** | NetworkX (`networkx`) — used to build directed graphs from Terraform configurations |
| **IaC Parser** | `python-hcl2` — parses HashiCorp Configuration Language (`.tf` files) into Python dictionaries |
| **Static Analysis / Labeling** | Checkov — scans Terraform repositories for security misconfigurations and generates `checkov_report.json` per repo |
| **Evaluation Metrics** | scikit-learn (`sklearn`) — `f1_score` (macro average), `confusion_matrix` |
| **Numerical Computing** | NumPy — class weight computation, array operations |
| **Serialization** | `pickle` — used to serialize/deserialize NetworkX graphs (`.gpickle` format) and node-type vocabulary maps |
| **Parallelism** | `concurrent.futures.ThreadPoolExecutor` (4 workers) — parallel Checkov scanning across repositories |
| **Hardware** | CPU/GPU — model supports CUDA-enabled GPU via `torch.device` |

---

### 2. Feature Extraction

The feature extraction pipeline transforms raw Terraform IaC repositories into graph-structured data suitable for the RGCN model. It consists of the following stages:

#### Stage 1: Vulnerability Labeling (`label_dataset.py`)

- Each Terraform repository is scanned using **Checkov** (a static analysis tool for IaC) with the `--framework terraform` flag.
- Checkov performs a **full scan** (all built-in checks) and outputs results as `checkov_report.json` in each repository folder.
- Repositories are scanned in **parallel** using 4 threads for efficiency. Already-scanned repos are skipped on re-runs.

#### Stage 2: Graph Construction (`build_graphs.py`)

**Node Extraction:**
- Each `.tf` file in a repository is parsed using `hcl2`.
- Every `resource` block (e.g., `aws_s3_bucket`, `aws_iam_role`) becomes a **node** in a directed graph.
- Each node stores:
  - `type` — the normalized AWS resource type (e.g., `aws_s3_bucket`)
  - `config` — the raw resource configuration block
  - `risk_score` — the vulnerability label (0 = Safe, 1 = Low, 2 = Medium, 3 = High)

**Risk Score Assignment:**
- The `checkov_report.json` is parsed to extract **failed checks** per resource.
- Each failed check is assigned a risk score using:
  1. **Explicit severity** from Checkov output (CRITICAL/HIGH → 3, MEDIUM → 2, LOW → 1)
  2. **Heuristic keyword matching** on check names (e.g., `"0.0.0.0"`, `"public"` → High; `"logging"`, `"backup"` → Medium; `"tag"` → Low)
  3. **Default** → Medium (score 2) if no match is found
- For resources with multiple failed checks, the **maximum risk score** is kept.
- Resources with **no failed checks** are labeled as Safe (score 0).

**Edge Construction:**
- Directed edges are created between nodes using two heuristics:
  1. **Direct reference** — source resource's raw config contains the target's Checkov-style ID (`resource_type.resource_name`)
  2. **Token overlap** — fuzzy matching based on shared unique tokens extracted from resource names (with stop-word filtering and minimum token length of 4)
- Edge types are assigned as:
  - `permission` (type 1) — if either the source or target involves a security-related resource (IAM, policy, role, security group, ACL)
  - `dependency` (type 0) — all other edges

**Output:** Graphs with valid nodes and edges are serialized as `.gpickle` files (up to 1,500 repos).

#### Stage 3: PyTorch Geometric Conversion (`cloud_dataset.py`)

- All `.gpickle` files are loaded and converted to `torch_geometric.data.Data` objects.
- **Node feature (`x`):** A global vocabulary (`node_type_map`) is built from all unique AWS resource types across the dataset. Each node's feature is the **integer index** of its resource type in this vocabulary — a single categorical feature per node (no continuous features).
- **Node label (`y`):** The `risk_score` attribute (0–3) serves as the classification target.
- **Edge index:** Node IDs are re-mapped to contiguous indices `0...N-1`. Edges are stored as a `[2, E]` tensor.
- **Edge type:** Each edge is typed as `0` (dependency) or `1` (permission).
- The processed dataset is saved as `data.pt` (graph data + slices) and `node_type_map.pkl` (vocabulary mapping) for efficient reloading via `InMemoryDataset`.

---

### 3. Training (`rgcn_train.py`)

#### Data Splitting

- The full dataset is **shuffled** randomly and split into:
  - **Train:** 80% — used for gradient updates
  - **Validation:** 10% — used for model selection (best checkpoint)
  - **Test:** 10% — used for final evaluation (loaded from checkpoint)
- A single random split is used (no k-fold cross-validation).

#### Model Initialization

- The RGCN model is instantiated with:
  - `num_node_types` = vocabulary size + 1 (for unseen types)
  - `hidden_channels` = 64
  - `num_classes` = 4 (Safe, Low, Medium, High)
  - `num_relations` = 2 (dependency, permission)

#### Class Imbalance Handling

- Class frequencies are computed from the **training set** node labels.
- **Inverse class-frequency weights** are applied: `weight[c] = total_samples / (num_classes × count[c])`
- These weights are passed to `CrossEntropyLoss` to penalize misclassification of minority risk classes.

#### Optimization

| Parameter | Value |
|---|---|
| **Optimizer** | Adam (`torch.optim.Adam`) |
| **Learning Rate** | 0.01 |
| **Loss Function** | `CrossEntropyLoss` with inverse-frequency class weights |
| **Epochs** | 50 |
| **Batch Size** | 32 (via `DataLoader`) |

#### Training Loop (per epoch)

1. **Forward pass:** Batched graphs are fed through the RGCN; loss is computed on **all nodes** in the batch.
2. **Backward pass:** Gradients are computed and parameters updated via Adam.
3. **Validation:** After each epoch, the model is evaluated on the validation set:
   - **Accuracy** = correct node predictions / total nodes
   - **Macro F1 Score** = `sklearn.metrics.f1_score` with `average='macro'`
4. **Model Checkpointing:** If the current validation accuracy exceeds the best so far, the model state dict is saved to `rgcn_model.pth`.

#### Final Evaluation

- After training completes, the **best checkpoint** (`rgcn_model.pth`) is loaded.
- The model is evaluated on the held-out **test set**, reporting:
  - Test Accuracy
  - Test Macro F1 Score
  - Confusion Matrix (via `sklearn.metrics.confusion_matrix`)

---

## ⚠️ Potential Observations / Improvements

1. **Sparse node features** — Only a single categorical feature (resource type) is used. Adding more features (e.g., config attributes, number of connections) could improve performance.
2. **No cross-validation** — A single 80/10/10 split; k-fold CV would be more robust.
3. **No precision/recall separately** — Only macro F1 is reported; per-class precision/recall would give deeper insight with class imbalance.
4. **Class imbalance handling** — ✅ Already addressed with inverse-frequency class weights in the loss function.
5. **`device` variable bug** — In `rgcn_train.py`, `device` is used but never defined. It should be `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` with `model.to(device)`.

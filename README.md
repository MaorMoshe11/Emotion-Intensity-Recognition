# Emotion Intensity Recognition (CBM)

Predict **Interest** from text via interpretable **Valence** & **Arousal** concepts using **Concept Bottleneck Models (CBMs)**. The pipeline also tackles **missingness** (correlation‑preserving imputation) and **class imbalance** (LIC‑based contrastive augmentation). Primary results are reported on a **clean** test split (no augmented sentences).

---

## TL;DR

* **Task:** Predict 1–7 ordinal labels for *Interest* from text.
* **Concepts:**

  * Interpretable bottleneck of **Valence** & **Arousal** (1–7 each).
  * Sequential CBM: learn concepts → predict target.
* **Data issues handled:** large missingness; severe class imbalance.
* **Fixes:** correlation‑preserving imputation; LIC‑guided synonym augmentation; careful evaluation (no aug in test).
* **Backbones:** `bert_cls_embedding`, `bert_mean_embedding`, `sen_embedding_qwen0.6b` (best on clean split).
* **Metrics:** macro/weighted F1, accuracy (weighted), confusion matrices (ordinal error focus).
* **Key finding:** A 2‑D affect bottleneck is **too narrow** to fully recover *Interest*; widening the concept set is promising.

---

## 1) Motivation & Overview

Human affect can be organized along **Valence** (positive↔negative) and **Arousal** (low↔high). We study whether these two dimensions suffice as an interpretable **bottleneck** to predict perceived **Interest** in short text segments. We combine:

1. **Correlation‑preserving imputation** to respect emotion–emotion structure.
2. **LIC‑based contrastive augmentation** to upweight minority classes without breaking semantics.
3. **Concept Bottleneck Models** for transparency: embeddings → (Valence, Arousal) → Interest.

We emphasize **clean evaluation**: test sets exclude augmented sentences to avoid semantic leakage.

---

## 2) Method Summary

### 2.1 Correlation‑Preserving Imputation

* Preserve the empirical correlation map among emotions when filling missing labels.
* Round to nearest integer (1–7) after imputation to keep ordinal labels consistent.

### 2.2 LIC‑Based Contrastive Augmentation

* Compute a **Label‑Indicative Coefficient (LIC)** per word & class.
* For minority classes, create **positive contrastive** variants by replacing high‑LIC tokens with near‑synonyms (Word2Vec‑style neighbors).
* **Important:** Augmented sentences are **never** included in the **test** split used for primary reporting.

### 2.3 Concept Bottleneck Model (CBM)

* **Inputs:** sentence embeddings (BERT CLS/MEAN; Qwen 0.6B sentence embeddings).
* **Concept head** $g_\theta$: shared MLP → two heads (Valence, Arousal), each 7‑way CE loss.
* **Target head** $f_\phi$: MLP on concatenated concept logits → Interest (1–7) via CE.
* **Training:** sequential (train concepts → freeze → train target).

> **Pipeline figure:** `figs/pipeline.png`

---

## 3) Results (Clean Test Split)

On the clean test set (no augmented sentences), the **Qwen 0.6B sentence‑embedding backbone** provides the strongest F1 across targets.

> **Figure:** Confusion matrices for Arousal, Valence, Interest → `figs/cm_sen_embeddings.png`

**Typical error pattern:** Adjacent‑class confusions in the mid‑range (3–5), consistent with ordinal ambiguity. Extremes (1, 7) are better separated.

> **Table:** Embedding comparison — macro/weighted F1 → `tables/embed_f1_clean.csv`

**Caveat:** If augmented sentences are added to test, F1 inflates due to **semantic leakage** (train–test near‑duplicates in embedding space). We therefore report primary results on the clean split.

---

## 4) Limitations & Next Steps

* **Data:** Only **12 participants** → limited speaker diversity; cross‑subject generalization is weak. Collect broader, more diverse speakers.
* **Modeling:** CrossEntropy treats all mistakes equally; MSE ignores ordinality. Explore **ordinal regression** losses/architectures (e.g., CORN/CORAL‑style) to penalize large ordinal errors more than adjacent ones.
* **Concept space:** **Valence + Arousal** are often **insufficient** for Interest. Consider **wider bottlenecks** (e.g., six basic emotions, or VAD with Dominance) while retaining interpretability.

---

## 5) Repo Structure

```
emotion-intensity-recognition/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/                     # original CSVs / JSONL
│  ├─ processed/               # post-imputation, splits
│  └─ metadata/                # mapping tables, label info
├─ figs/
│  ├─ pipeline.png
│  ├─ cm_sen_embeddings.png
│  ├─ imputation_errors.png
│  └─ ...
├─ tables/
│  └─ embed_f1_clean.csv
├─ src/
│  ├─ data/
│  │  ├─ impute.py             # correlation-preserving imputer
│  │  ├─ augment_lic.py        # LIC scoring & augmentation
│  │  └─ splits.py             # clean train/val/test (no aug in test)
│  ├─ models/
│  │  ├─ cbm.py                # g_theta (concepts), f_phi (target)
│  │  ├─ heads.py              # MLP heads
│  │  └─ losses.py             # CE, ordinal variants (future)
│  ├─ train/
│  │  ├─ train_concepts.py     # stage 1 training
│  │  └─ train_interest.py     # stage 2 training
│  ├─ eval/
│  │  ├─ metrics.py            # macro/weighted F1, weighted acc
│  │  ├─ confusion.py          # CMs, reliability/ECE (optional)
│  │  └─ report.py             # aggregate tables & plots
│  └─ utils/
│     ├─ seed.py               # deterministic seeds
│     └─ io.py                 # I/O helpers
└─ scripts/
   ├─ 00_prepare_data.sh
   ├─ 01_impute.py
   ├─ 02_augment_lic.py
   ├─ 03_make_splits.py
   ├─ 10_train_concepts.py
   ├─ 11_train_interest.py
   └─ 20_eval_report.py
```

---

## 6) Quickstart

### 6.1 Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Data Preparation

```bash
python scripts/01_impute.py \
  --in data/raw/emotions.csv \
  --out data/processed/emotions_imputed.csv \
  --method correlation

python scripts/02_augment_lic.py \
  --in data/processed/emotions_imputed.csv \
  --out data/processed/emotions_aug.csv \
  --w2v path/to/word2vec.bin \
  --max_ratio 2.0

python scripts/03_make_splits.py \
  --in data/processed/emotions_aug.csv \
  --out data/processed/splits \
  --no_aug_in_test
```

### 6.3 Training (Sequential CBM)

```bash
# Stage 1: learn concepts (Valence, Arousal)
python scripts/10_train_concepts.py \
  --emb_col sen_embedding_qwen0.6b \
  --train data/processed/splits/train.csv \
  --val   data/processed/splits/val.csv \
  --out   runs/concepts_qwen/

# Stage 2: predict Interest from concept logits
python scripts/11_train_interest.py \
  --concept_ckpt runs/concepts_qwen/best.pt \
  --train data/processed/splits/train.csv \
  --val   data/processed/splits/val.csv \
  --out   runs/interest_qwen/
```

### 6.4 Evaluation & Reports

```bash
python scripts/20_eval_report.py \
  --ckpt runs/interest_qwen/best.pt \
  --test data/processed/splits/test_clean.csv \
  --out  reports/qwen_clean/
```

Outputs include macro/weighted F1, weighted accuracy, and confusion matrices. Reliability/ECE plots are optional.

---

## 7) Reproducibility Notes

* **Seeds:** fixed seeds for NumPy/PyTorch; report fold means/stdevs.
* **No Aug in Test:** primary results *must* use the clean test split (no augmented sentences).
* **Rounded comparison:** when comparing to regressors, round to 1–7 and clip to legal range for classification metrics.

---

## 8) Results Snapshot (placeholders)

| Embedding / Setting                   | Arousal F1 (mac/w) | Valence F1 (mac/w) | Interest F1 (mac/w) |
| ------------------------------------- | ------------------ | ------------------ | ------------------- |
| bert\_cls\_embedding                  | 0.515 / 0.454      | 0.489 / 0.418      | 0.305 / 0.304       |
| bert\_mean\_embedding                 | 0.484 / 0.459      | 0.476 / 0.411      | 0.290 / 0.302       |
| sen\_embedding\_qwen0.6b (clean)      | **0.515 / 0.461**  | **0.493 / 0.431**  | **0.317 / 0.325**   |
| sen\_embedding\_qwen0.6b (Aug Test)\* | 0.810 / 0.746      | 0.785 / 0.741      | 0.675 / 0.682       |

\* Inflated by semantic leakage; report clean scores as primary.

---

## 9) Citation

If you use this code or ideas, please cite the project/paper:

```bibtex
@misc{moshe2025emotioncbm,
  title   = {Emotion Intensity Recognition via Concept Bottlenecks},
  author  = {Maor Moshe and Idan Salomon and David Oriel},
  year    = {2025},
  note    = {GitHub repository},
  url     = {https://github.com/your/repo}
}
```

---

## 10) License

MIT (or update accordingly).

---

## 11) Acknowledgments

Thanks to collaborators and advisors for discussion and feedback. Replace this section with your specific acknowledgments.

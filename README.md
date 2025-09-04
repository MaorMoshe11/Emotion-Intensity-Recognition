# Emotion Intensity Recognition (CBM)

Predict **Interest** from text using interpretable affective concepts (**Valence** & **Arousal**) via **Concept Bottleneck Models (CBMs)**.  
The pipeline addresses **missingness** with correlation-preserving imputation and **class imbalance** with LIC-based contrastive augmentation.  
All results are reported on a **clean** test split (no augmented sentences).

---

## 📂 Repository Contents

- **EDA_EIR.ipynb** – Exploratory data analysis: distributions, missingness, correlations.  
- **NA_Treatment.ipynb** – Correlation-preserving imputation and comparison of imputation methods.  
- **Text Augmentation.ipynb** – LIC-based synonym augmentation for minority classes.  
- **Basic_Models.py** – Baseline models (Lasso, Ridge, XGBoost) for Valence, Arousal, Interest.  
- **Advenced_Model_For_EIR.ipynb** – CBM pipeline: sentence embeddings → (Valence, Arousal) → Interest.  
- **Emotional Intensity Recognition Paper.pdf** – Full project report with methodology, results, and analysis.  
- **EIRVideo.mp4** – Demo video presentation of the project.  
- **README.md** – Project overview (this file).

---

## 🔑 Key Ideas

1. **Correlation-Preserving Imputation**  
   Maintains the original emotion–emotion correlation map when filling missing values, preventing distortion of structure.

2. **LIC-Based Contrastive Augmentation**  
   Uses Label-Indicative Coefficients to generate minority-class variants by synonym replacement, balancing Valence/Arousal classes.

3. **Concept Bottleneck Models (CBMs)**  
   Pipeline:  
   - Sentence embeddings (BERT CLS/Mean, Qwen 0.6B).  
   - Concept head → predict Valence & Arousal (7-class each).  
   - Target head → predict Interest (1–7).  
   - Sequential training (concepts first, then target).  

4. **Evaluation**  
   - Weighted/macro F1 to handle imbalance.  
   - Confusion matrices to expose ordinal error trends.  
   - Strictly *clean* test sets (no augmented samples).  

---

## 📊 Results Snapshot (Clean Test)

| Embedding                  | Arousal F1 (mac/w) | Valence F1 (mac/w) | Interest F1 (mac/w) |
|-----------------------------|--------------------|--------------------|---------------------|
| bert_cls_embedding          | 0.515 / 0.454      | 0.489 / 0.418      | 0.305 / 0.304       |
| bert_mean_embedding         | 0.484 / 0.459      | 0.476 / 0.411      | 0.290 / 0.302       |
| sen_embedding_qwen0.6b      | **0.515 / 0.461**  | **0.493 / 0.431**  | **0.317 / 0.325**   |

⚠️ Including augmented sentences in the test set artificially inflates F1 (semantic leakage). Only clean-split results are valid.

---

## 🚧 Limitations & Future Work

- **Data**: Only 12 participants → limited stylistic diversity. Future work should expand to larger and more varied populations.  
- **Modeling**: CE treats all misclassifications equally, MSE ignores ordinality. Ordinal regression losses (CORN/CORAL) are promising.  
- **Concept Space**: 2D Valence+Arousal is often too narrow. Widening the bottleneck (e.g., basic emotions) can improve expressivity while staying interpretable.  

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@misc{moshe2025emotioncbm,
  title   = {Emotion Intensity Recognition via Concept Bottlenecks},
  author  = {Maor Moshe and Idan Salomon and David Oriel},
  year    = {2025},
  note    = {GitHub repository},
  url     = {https://github.com/MaorMoshe11/Emotion-Intensity-Recognition}
}

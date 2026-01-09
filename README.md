# 📰 Political News Bias Detection & Clustering

## 📌 Overview
This project analyzes **political news bias** using Natural Language Processing (NLP).  
It has two main components:  

1. **Bias Detection (Classification):** Classify political news articles into categories such as *Left / Center / Right*.  
2. **Bias-Aware Clustering:** Group related news articles using semantic embeddings while incorporating political bias awareness.  

The goal is to show how different outlets report the same story with different political leanings.

---

## ⚙️ Tech Stack
- **Python, PyTorch**
- **Hugging Face Transformers (RoBERTa / DeBERTa)** for bias detection  
- **Sentence-Transformers** for semantic embeddings  
- **Scikit-learn** for clustering and evaluation  
- **Pandas, NumPy** for preprocessing  
- **Matplotlib/Seaborn** for visualization  
- **Jupyter/Colab** for experimentation  

---

## 🚀 Features
- ✅ Preprocess articles (clean, normalize, remove URLs)  
- ✅ Train supervised **bias classifier** (Left / Center / Right)  
- ✅ Generate embeddings with **RoBERTa/DeBERTa**  
- ✅ Cluster articles by semantic similarity with **cosine distance**  
- ✅ **Bias-aware clustering**: boost similarity for same-bias articles  
- ✅ Visualize results (confusion matrix, cluster summaries, bias distribution)  

---
## 📊 Results
- **Bias Classifier:** Achieved ~86% accuracy on labeled political news dataset  
- **Clustering:** Groups semantically related articles into coherent clusters  
- **Bias-Aware Boost:** Articles with same leaning cluster more strongly  


---
## 📊 Results

### 🔹 Training Performance
| Epoch | Training Loss | Validation Loss | Accuracy | F1 Macro | F1 Weighted |
|-------|---------------|-----------------|----------|----------|-------------|
| 1     | 0.7302        | 0.4661          | 0.8007   | 0.7949   | 0.8012      |
| 2     | 0.3797        | 0.3906          | 0.8495   | 0.8424   | 0.8486      |
| 3     | 0.2848        | 0.4358          | 0.8468   | 0.8403   | 0.8458      |
| 4     | 0.1981        | 0.4552          | 0.8556   | 0.8492   | 0.8551      |

---

### 🔹 Sample Predictions (Validation Set)
<details>
<summary>Click to expand</summary>

**Sample #1**  
📜 *Text (truncated)*: *Roger Ailes … Fox News into a media juggernaut …*  
✅ Predicted: **Left**  
🎯 Actual: **Left**

---

**Sample #2**  
📜 *Text (truncated)*: *Mike Huckabee called Friday's shootings …*  
✅ Predicted: **Left**  
🎯 Actual: **Left**

---

**Sample #3**  
📜 *Text (truncated)*: *Job growth slowed dramatically in March …*  
✅ Predicted: **Left**  
🎯 Actual: **Left**

---

**Sample #4**  
📜 *Text (truncated)*: *President Biden called a new Texas voting overhaul …*  
✅ Predicted: **Right**  
🎯 Actual: **Right**

---

**Sample #5**  
📜 *Text (truncated)*: *U.S. District Judge ruled the Deferred Action …*  
✅ Predicted: **Center**  
🎯 Actual: **Center**

---
</details>

---

### 🔹 Additional Test Predictions
<details>
<summary>Click to expand</summary>

**Article #1**  
📜 *Text*: *The government's crackdown on climate activists …*  
🔍 Predicted Bias: **Left**

---

**Article #2**  
📜 *Text*: *With income inequality at historic highs …*  
🔍 Predicted Bias: **Right**

---

**Article #3**  
📜 *Text*: *The Senate has passed a bipartisan infrastructure bill …*  
🔍 Predicted Bias: **Right**

---

**Article #4**  
📜 *Text*: *Experts are urging caution on foreign policy moves …*  
🔍 Predicted Bias: **Left**

---

**Article #5**  
📜 *Text*: *A new wave of conservative leaders … stronger border security …*  
🔍 Predicted Bias: **Right**

---
</details>

## 📌 Future Work
- 🔄 Replace MiniLM with **RoBERTa / DeBERTa embeddings** for richer context  
- 📈 Improve detection with larger labeled datasets  
- 🌐 Deploy as an API for real-time bias monitoring  
- 📊 Build a dashboard to visualize clusters & bias distributions  

---

## ✨ Credits
- [Sentence-Transformers](https://www.sbert.net/)  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [RoBERTa](https://huggingface.co/roberta-base)  
- [DeBERTa](https://huggingface.co/microsoft/deberta-v3-base)  

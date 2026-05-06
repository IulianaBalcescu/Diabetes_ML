# 🩺 Diabetes Prediction ML Model

[![Python 3.13](https://img.shields.io/badge/Python-3.13.9-blue?style=flat&logo=python)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](#license)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)]()

> A machine learning project that builds a **Decision Tree Classifier** to predict diabetes risk based on patient health metrics. This repository demonstrates data preprocessing, model training, evaluation, and optimization techniques for medical prediction tasks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [What Does This Project Do?](#what-does-this-project-do)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage & Examples](#usage--examples)
- [Model Performance](#model-performance)
- [Screenshots & Visualizations](#screenshots--visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Questions & Support](#questions--support)

---

## 🎯 Overview

This project implements a **Decision Tree-based machine learning model** to predict diabetes risk. It uses the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and demonstrates:

✅ **Data exploration & cleaning** - Handling missing/zero values in medical data  
✅ **Model training & evaluation** - Using accuracy, recall, and confusion matrices  
✅ **Model optimization** - Improving predictions with class balancing and threshold tuning  
✅ **Visualization** - Exporting decision trees for interpretability  

---

## 🔬 What Does This Project Do?

### The Problem
Diabetes is a serious health condition. Early prediction can help with preventive care. This ML model learns from historical patient data to predict whether a new patient is at risk of diabetes.

### The Solution
- **Input**: 8 health metrics (glucose, blood pressure, BMI, etc.)
- **Output**: Binary prediction (Diabetes: Yes/No)
- **Approach**: Decision Tree Classifier with optimizations
- **Focus**: High Recall to minimize missed cases ⚠️

### Key Features
| Feature | Description |
|---------|-------------|
| **Data Cleaning** | Replaces suspicious zero values with median values |
| **Class Balancing** | Uses `class_weight="balanced"` to give equal importance to both classes |
| **Threshold Tuning** | Adjustable probability threshold (default 0.5, can be set to 0.4) to reduce false negatives |
| **Exportable Trees** | Generates `.dot` files for visualizing the decision logic |

---

## 🛠 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.13.9** | Programming language |
| **Jupyter Notebook** | Interactive development & visualization |
| **pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **scikit-learn** | Machine learning algorithms & metrics |
| **Graphviz** | Decision tree visualization |

```python
# Main Libraries Used
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
```

---

## 📁 Project Structure

```
Diabetes_ML/
│
├── README.md                              # 📄 This file
├── ML/
│   ├── Diabetes.ipynb                    # 🔥 Main Jupyter Notebook (STARTS HERE)
│   ├── diabetes.csv                      # 📊 Dataset (768 patients, 9 columns)
│   ├── diabetes_tree_clean.dot           # 🌳 Exported decision tree
│   ├── Diabetes_ML_Documentation_EN.docx # 📖 Detailed English documentation
│   └── Documentatie_Diabetes_ML_RO.docx  # 📖 Detailed Romanian documentation
│
```

### 📊 Dataset Overview

**File**: `ML/diabetes.csv`  
**Rows**: 768 patient records  
**Columns**: 9 features

| Column Name | Description | Range | Unit |
|-------------|-------------|-------|------|
| Pregnancies | Number of pregnancies | 0-17 | count |
| Glucose | Fasting blood glucose level | 0-199 | mg/dL |
| BloodPressure | Diastolic blood pressure | 0-122 | mmHg |
| SkinThickness | Triceps skinfold thickness | 0-99 | mm |
| Insulin | 2-hour serum insulin | 0-846 | mu U/ml |
| BMI | Body Mass Index | 0-67 | kg/m² |
| DiabetesPedigreeFunction | Genetic predisposition score | 0.0-2.4 | ratio |
| Age | Patient age | 21-81 | years |
| **Outcome** | Target: Diabetes present (1) or not (0) | 0 or 1 | binary |

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.7 or higher (tested with 3.13.9)
- pip or conda package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository

```bash
git clone https://github.com/IulianaBalcescu/Diabetes_ML.git
cd Diabetes_ML
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n diabetes_ml python=3.13
conda activate diabetes_ml
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy scikit-learn jupyter notebook
```

Or from a requirements file (if you create one):

```bash
pip install -r requirements.txt
```

**Suggested `requirements.txt`:**
```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.0
jupyter==1.0.0
notebook==6.5.0
graphviz==0.20.1
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open `ML/Diabetes.ipynb` in your browser.

---

## 🚀 Usage & Examples

### Quick Start

1. **Open the notebook**: `ML/Diabetes.ipynb`
2. **Run all cells** (Kernel → Restart & Run All)
3. **Review outputs** in the notebook

### Understanding the Notebook Structure

<details>
<summary><b>📝 Cell 1: Data Loading & Exploration</b></summary>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv('diabetes.csv')

# Quick inspection
data.shape          # (768, 9)
data.head()         # First 5 rows
data.describe()     # Statistics
data.isnull().sum() # Check for null values
```

**Output**: Basic dataset statistics

</details>

<details>
<summary><b>🔧 Cell 2: Train Initial Model (No Cleaning)</b></summary>

```python
# Split data: 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create & train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# Evaluate
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
```

**Output**:
- Accuracy: ~74.7%
- Recall: ~72.7%

</details>

<details>
<summary><b>🧹 Cell 3: Data Cleaning & Improved Model</b></summary>

The dataset has suspicious zero values (e.g., glucose=0 is impossible). This cell fixes them:

```python
# Replace zero values with median for these columns
columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for column in columns_to_fix:
    median_value = data_clean[column].replace(0, np.nan).median()
    data_clean[column] = data_clean[column].replace(0, median_value)
```

**Improvements**:
- Better data quality
- Higher recall (more diabetes cases caught)

</details>

<details>
<summary><b>🎯 Cell 4: Threshold Tuning</b></summary>

```python
# Instead of default 0.5 threshold, use 0.4
probabilities = model_clean.predict_proba(x_test_clean)[:, 1]

predictions_threshold = []
for prob in probabilities:
    if prob >= 0.4:  # More sensitive threshold
        predictions_threshold.append(1)
    else:
        predictions_threshold.append(0)
```

**Benefit**: Catches more diabetes cases (higher recall) but may have more false alarms.

</details>

<details>
<summary><b>🌳 Cell 5: Export Decision Tree</b></summary>

```python
from sklearn import tree

tree.export_graphviz(
    model_clean,
    out_file="diabetes_tree_clean.dot",
    feature_names=x_clean.columns,
    class_names=["No Diabetes", "Diabetes"],
    label="all",
    rounded=True,
    filled=True
)
```

**Output**: `diabetes_tree_clean.dot` file (visualizable with Graphviz)

</details>

### Example: Making Predictions

```python
# Predict for a new patient
new_patient = pd.DataFrame([[
    Pregnancies=2, 
    Glucose=120,      # Slightly elevated
    BloodPressure=70,
    SkinThickness=20,
    Insulin=80,
    BMI=25.0,
    DiabetesPedigreeFunction=0.5,
    Age=30
]], columns=x.columns)

prediction = model_clean.predict(new_patient)

if prediction[0] == 1:
    print("⚠️ Model predicts: DIABETES RISK")
else:
    print("✅ Model predicts: NO DIABETES RISK")
```

---

## 📊 Model Performance

### Initial Model (No Data Cleaning)

| Metric | Value |
|--------|-------|
| **Accuracy** | 74.7% |
| **Recall** | 72.7% |
| **Confusion Matrix** | [[75, 24], [15, 40]] |

### Optimized Model (Data Cleaned + Balanced)

| Metric | Value |
|--------|-------|
| **Accuracy** | 70.1% |
| **Recall** | 81.8% ⬆️ |
| **Confusion Matrix** | [[63, 36], [10, 45]] |

### With Threshold = 0.4 (Medical-Focused)

| Metric | Value |
|--------|-------|
| **Accuracy** | 70.1% |
| **Recall** | 81.8% ✅ |
| **False Negatives** | 10 (fewer missed cases!) |

**Note**: Recall is more important in medical diagnosis to avoid missing diabetic patients! 🏥

---

## 🖼 Screenshots & Visualizations

### 📊 TODO: Add Visualizations

Suggested additions for future versions:

```markdown
#### Decision Tree Visualization
![Decision Tree](./images/diabetes_tree.png)
*Generated from diabetes_tree_clean.dot using Graphviz*

#### Confusion Matrix Heatmap
![Confusion Matrix](./images/confusion_matrix.png)

#### Feature Importance
![Feature Importance](./images/feature_importance.png)

#### Model Comparison
![Model Comparison](./images/model_metrics_comparison.png)
```

**How to generate tree visualization locally**:
```bash
# Install Graphviz
pip install graphviz

# Convert .dot to PNG
dot -Tpng ML/diabetes_tree_clean.dot -o diabetes_tree_clean.png
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### How to Contribute

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/your-idea`
3. **Make changes** and test thoroughly
4. **Commit** with clear messages: `git commit -m "Add: improved model accuracy"`
5. **Push** to your fork: `git push origin feature/your-idea`
6. **Open a Pull Request** with description

### Areas for Improvement

- [ ] **Improve model accuracy** (try Random Forest, SVM, or Neural Networks)
- [ ] **Add cross-validation** for more robust evaluation
- [ ] **Create visualizations** (confusion matrix, ROC curves, etc.)
- [ ] **Add unit tests** for reproducibility
- [ ] **Build a Flask/Streamlit app** for interactive predictions
- [ ] **Document medical insights** from the decision tree
- [ ] **Add more features** or use external datasets

### Code Style

- Follow PEP 8 conventions
- Use descriptive variable names
- Add comments for complex logic
- Keep cells focused on one task in Jupyter

---

## 📄 License

This project is licensed under the **MIT License** - feel free to use, modify, and distribute this code.

See [LICENSE](./LICENSE) file for details, or see standard MIT terms:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ❌ Liability
- ❌ Warranty

---

## ❓ Questions & Support

### Getting Help

- 📖 **Read the docs**: Check `ML/Diabetes_ML_Documentation_EN.docx` (English) or `ML/Documentatie_Diabetes_ML_RO.docx` (Romanian)
- 💬 **Open an Issue**: [GitHub Issues](https://github.com/IulianaBalcescu/Diabetes_ML/issues)
- 📧 **Contact**: [GitHub Profile](https://github.com/IulianaBalcescu)

### FAQ

<details>
<summary><b>Q: Why is accuracy lower after data cleaning?</b></summary>

**A**: Because we're being more conservative with predictions (higher recall). We're catching MORE diabetes cases, which slightly reduces overall accuracy but is better for medical diagnosis. It's a trade-off: fewer false negatives (missed cases) vs. more false positives (unnecessary alerts).

</details>

<details>
<summary><b>Q: What do the zero values in the dataset mean?</b></summary>

**A**: The dataset has suspicious zeros that likely represent missing data:
- Glucose = 0 is impossible (minimum viable level for living humans)
- BloodPressure = 0 same issue
- These are replaced with median values of the column for more realistic training

</details>

<details>
<summary><b>Q: How do I visualize the decision tree?</b></summary>

**A**: Use Graphviz:
```bash
pip install graphviz
dot -Tpng ML/diabetes_tree_clean.dot -o tree.png
```

Or online tools like [Graphviz Online](https://dreampuf.github.io/GraphvizOnline/)

</details>

<details>
<summary><b>Q: Can I use this model for real medical diagnosis?</b></summary>

**A**: ⚠️ **NO** - This is an educational project. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.

</details>

---

## 📚 Learn More

- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Machine Learning Metrics Explained](https://towardsdatascience.com/accuracy-recall-precision-f-score-explained-449d9a2d7f2c)
- [Medical ML Ethics](https://www.nature.com/articles/d41586-019-02807-z)

---

## 👨‍💻 Author

**Iuliana Balcescu**  
🔗 [GitHub](https://github.com/IulianaBalcescu) | 📧 [Contact](https://github.com/IulianaBalcescu)

---

## 🙌 Acknowledgments

- Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- Built with ❤️ for healthcare ML education
- Inspired by real-world medical data analysis challenges

---

<div align="center">

### ⭐ If you found this useful, please consider giving it a star!

**Made with 💙 for everyone interested in Machine Learning**

</div>

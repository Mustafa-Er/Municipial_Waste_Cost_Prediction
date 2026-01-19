# 🏙️ Municipal Waste Management Cost Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/shashwatwork/municipal-waste-management-cost-prediction)

---

## 📋 Project Overview

This project predicts **municipal solid waste (MSW) management costs** for Italian municipalities using machine learning and deep learning techniques. The study rigorously compares classical regression models against deep learning architectures within a **K-Fold Cross-Validation** framework.

> **Key Finding**: Simple linear models can outperform complex neural networks on structured tabular data! 🎯

---

## 📊 Dataset

**Source**: [Kaggle - Municipal Waste Management Cost Prediction](https://www.kaggle.com/datasets/shashwatwork/municipal-waste-management-cost-prediction)

| **Attribute** | **Details** |
|---------------|-------------|
| 📍 Municipalities | 4,341 Italian cities |
| 🎯 Target Variable | `msw` (Municipal Solid Waste in kg) |
| 📈 Input Features | 18 features: population, area, altitude, urbanization index, waste composition |

### 🗂️ Key Features

- **Demographic**: Population (`pop`), Area (`area`), Altitude (`alt`)
- **Urban Metrics**: Urbanization index (`urb`)
- **Waste Composition**: Organic, Paper, Plastic, Glass percentages

---

## 🔬 Methodology & Pipeline

### 🛠️ Data Preprocessing
```
📥 Raw Data
   ↓
🔄 KNN Imputation (k=5) → Handle missing values
   ↓
📏 RobustScaler → Normalize features & target
   ↓
✅ Ready for Training
```

| **Step** | **Method** | **Purpose** |
|----------|-----------|-------------|
| **Imputation** | KNN Imputer (k=5) | Preserve local data structures |
| **Normalization** | RobustScaler | Minimize outlier impact |
| **Validation** | 6-Fold CV | Ensure generalizability |

---

### 🤖 Models Evaluated

<table>
  <tr>
    <td align="center">📐<br><b>Ridge Regression</b><br>L2 regularization</td>
    <td align="center">🔮<br><b>SVM</b><br>RBF kernel</td>
    <td align="center">🧠<br><b>MLP</b><br>2 hidden layers</td>
    <td align="center">🏗️<br><b>1D-CNN</b><br>VGG-inspired blocks</td>
  </tr>
</table>

#### Model Architectures

1. **Ridge Regression (LR)** 📐  
   - Linear model with L2 regularization
   - Fast and interpretable

2. **Support Vector Regression (SVM)** 🔮  
   - RBF kernel for non-linear relationships
   - Tested for complex pattern detection

3. **Multi-Layer Perceptron (MLP)** 🧠  
   - Dense neural network: 2 × 32-unit hidden layers
   - Adam optimizer

4. **1D-CNN** 🏗️  
   - VGG-style convolutional blocks
   - Batch Normalization + Max Pooling
   - Designed for feature pattern extraction

---

## 📈 Results & Discussion

### 🏆 Model Performance Comparison

| Model | R² Score | RMSE | Performance Notes |
|-------|----------|------|-------------------|
| **🥇 Ridge Regression (LR)** | **0.9999** | **110,292** | **Best Performance.** Extremely fast and accurate. |
| **🥈 MLP** | 0.9974 | 589,080 | High accuracy but computationally expensive. |
| **🥉 SVM** | 0.6549 | 7,165,693 | Failed to capture underlying patterns. |

---

### ⚠️ Note on CNN Model

The 1D-CNN architecture was **fully implemented** (4 convolutional blocks with VGG-style design) but **excluded from final evaluation**.

**Why?** 🤔
- **Computational Cost**: Training time was prohibitively high within 6-Fold CV
- **Diminishing Returns**: Ridge Regression achieved R² ≈ 0.9999 with fraction of resources
- **Conclusion**: For this tabular dataset with strong linear relationships, complex deep learning architectures were unnecessary

**Implementation Status**: ✅ Code available but commented out in final run

---

## 💡 Key Takeaways

> ### 🎓 "Simpler is Often Better"
> 
> For structured tabular data with clear linear relationships:
> - ✅ Ridge Regression outperformed complex neural networks
> - ✅ Lower computational cost = Faster deployment
> - ✅ Better interpretability for stakeholders

### 🔑 Success Factors

1. **📊 Data Quality**: RobustScaler + KNN Imputation stabilized linear models
2. **🎯 Feature Engineering**: Well-structured input features enabled simple models to excel
3. **⚖️ Model Selection**: Matching model complexity to data structure is crucial

---

## 🚀 Installation & Usage

### 📦 Install Dependencies
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

**Required Libraries:**
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)  
- scikit-learn (machine learning models)
- tensorflow/keras (deep learning models)

### 📥 Download the Dataset

1. Visit Kaggle: [Municipal Waste Management Dataset](https://www.kaggle.com/datasets/shashwatwork/municipal-waste-management-cost-prediction)
2. Download and extract `public_data_waste_fee.csv`

### ⚙️ Configuration

**Important**: Update the dataset path in your notebook!
```python
class config:
    dir_dataset = "/path/to/your/public_data_waste_fee.csv"  # 👈 Update this
```

### ▶️ Run the Analysis
```bash
jupyter notebook main.ipynb
```

The notebook will automatically:
- 📂 Load and preprocess data
- 🔄 Execute 6-Fold Cross-Validation
- 📊 Train and evaluate models (SVM, Ridge, MLP)
- 📈 Display performance metrics
---

## 🛠️ Technologies

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge" />
</p>

### 📚 Libraries Used

| Category | Tools |
|----------|-------|
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (SVM, Ridge, KFold, RobustScaler, KNN Imputer) |
| **Deep Learning** | TensorFlow/Keras (Sequential API for MLP and CNN) |

---

## 📝 Project Structure
```
municipal-waste-prediction/
│
├── main.ipynb                 # Main analysis notebook
├── public_data_waste_fee.csv  # Dataset (download separately)
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## 👤 Author

**Mustafa Er**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Mustafa-Er)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mustafa-er-483983146/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/aski1140)

---

## 📄 License

This project is open source and available for educational purposes.

---

<p align="center">
  <i>⭐ If you found this project helpful, please consider giving it a star!</i>
</p>

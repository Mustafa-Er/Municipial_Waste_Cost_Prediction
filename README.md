# Municipial_Waste_Cost_Prediction

🇮🇹 Municipal Waste Management Cost Prediction
1. Project Overview
This project aims to predict the municipal solid waste (MSW) management costs for various municipalities in Italy. By analyzing demographic, geographic, and waste composition data, the project evaluates multiple machine learning and deep learning approaches to find the most accurate and efficient predictive model.
A key focus of this study is the rigorous comparison between classical regression models (SVM, Ridge) and deep learning architectures (MLP, CNN) within a K-Fold Cross-Validation framework.
2. Dataset
The dataset consists of 4,341 Italian municipalities.
• Target Variable: msw (Municipal Solid Waste in kg).
• Input Features: 18 features including population (pop), area (area), altitude (alt), urbanization index (urb), and specific waste composition percentages (organic, paper, plastic, glass, etc.).
3. Methodology & Pipeline
Data Preprocessing
• Imputation: Missing values in waste composition columns were filled using KNN Imputer (k=5) to preserve local data structures.
• Normalization: A RobustScaler was applied to both input features and the target variable to minimize the impact of outliers, which are common in economic/demographic data.
• Validation Strategy: A 6-Fold Cross-Validation scheme was implemented to ensure the model's generalizability and prevent overfitting.
Modeled Architectures
Four distinct architectures were designed and implemented:
1. Ridge Regression (LR): A linear model with L2 regularization to prevent overfitting.
2. Support Vector Regression (SVM): Using an RBF kernel for non-linear relationships.
3. Multi-Layer Perceptron (MLP): A dense neural network with two hidden layers (32 units each) optimized with Adam.
4. 1D-CNN (Convolutional Neural Network): A deep learning model inspired by VGG blocks, featuring multiple 1D-Convolutional layers, Batch Normalization, and Max Pooling designed to capture patterns in the feature vector.
4. Results & Discussion
The models were evaluated based on R² Score and RMSE (Root Mean Squared Error). The results from the test set are summarized below:
| Model | R² Score | RMSE | Performance Notes |
| :--- | :---: | :---: | :--- |
| **Ridge Regression (LR)** | **0.9999** | **110,292** | **Best Performance.** Extremely fast and accurate. |
| **MLP** | 0.9974 | 589,080 | High accuracy but computationally more expensive. |
| **SVM** | 0.6549 | 7,165,693 | Failed to capture the underlying data structure. |

⚠️ Note on the CNN Model
Although a complex 1D-CNN architecture was fully implemented (containing 4 convolutional blocks and dense layers), it was excluded from the final results table.
• Reason: The computational cost and training time required for the CNN within a 6-Fold Cross-Validation loop were disproportionately high compared to the performance gains.
• Conclusion: Preliminary tests showed that simpler models like Ridge Regression provided superior accuracy (R 
2
 ≈1.0) with a fraction of the computational resources, demonstrating that for this specific tabular dataset, complex Deep Learning architectures were unnecessary.
5. Key Takeaways
• Model Selection: "Simpler is often better." The linear relationship in the data made Ridge Regression the optimal choice over complex Neural Networks.
• Data Quality: The use of RobustScaler and KNNImputer significantly contributed to the stability of the linear models.
6. Installation & Usage
# Clone the repository
git clone https://github.com/yourusername/municipal-waste-prediction.git

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
You can run the analysis by executing the Jupyter Notebook main.ipynb. The script will automatically handle data loading, preprocessing, and the K-Fold validation loop for the selected models.
7. Technologies
• Python
• Data Manipulation: Pandas, NumPy
• Visualization: Matplotlib, Seaborn
• Machine Learning: Scikit-learn (SVM, Ridge, KFold, RobustScaler)
• Deep Learning: TensorFlow/Keras (Sequential API for MLP and CNN)

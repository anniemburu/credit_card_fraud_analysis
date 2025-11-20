
# Credit Card Analysis : Supervised vs Unsupervised

## Overview

This document presents a detailed analysis of the performance of several machine learning models under different evaluation strategies. The primary goal is to identify the most effective models for a given task (implied to be a classification task based on the metrics) and understand how various evaluation approaches influence their reported performance.

## 🛠️ Setup and Installation
1. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other projects. You can choose between Python's built-in venv or Conda.

A. Using Conda
If you have Anaconda or Miniconda installed, use the following commands:

```
# Create a new environment named 'ml_env' with Python 3.9
conda create -n ml_env python=3.9

# Activate the environment
conda activate ml_env
```

B. Using Python's venv
Use the following commands from your terminal:

```
# Create a new environment named '.venv'
python3 -m venv .venv

# Activate the environment (macOS/Linux)
source .venv/bin/activate

# Activate the environment (Windows)
.\.venv\Scripts\activate
```

2. Install Dependencies
Once your virtual environment is active, install the required packages using the provided requirements.txt file.

```
# Install all required packages
pip install -r requirements.txt
```

## 🏃 Running the Project
This project uses MLflow for robust experiment tracking. All model runs, parameters, and metrics will be logged automatically.

1. Launch the MLflow Tracking Server
Before running the training script, start the MLflow server in the background. This will allow you to view the experiment results in your browser.

```
# Start MLflow server on default host/port (http://127.0.0.1:5000)
mlflow ui &
```

2. Run the Training Script
The main script for running the classification experiments is located at src.train.

To execute the project and log the results to the MLflow server, run on a different terminal:

```
# Run the main training file
python src.train
```

## 🔬 Viewing Experiment Results
After running the project, you can analyze the results:

    - Open your web browser and navigate to the MLflow UI address (usually http://127.0.0.1:5000).

    - Select your experiment 

    - You will see a list of runs, where you can compare the metrics (Accuracy, F1, ROC-AUC) and artifacts (like confusion matrices) for each model (LGBM, XGBoost, CatBoost)

##

```

├── src/
│   ├── __init__.py
│   ├── extract.py
│   ├── load.py
│   ├── transform
│   └── train.py         
├── requirements.txt  
└── README.md    
```

## Models Evaluated

The following machine learning models were included in this evaluation:

*   **LGBM (Light Gradient Boosting Machine)**
*   **XGBoost (Extreme Gradient Boosting)**
*   **CatBoost (CB)**
*   **Isolation Trees**

## Evaluation Strategies

Each model's performance was assessed using the following optimization strategies, which aim to prioritize different aspects of model behavior:

*   **F1 Score:** A measure that balances precision and recall, useful when an even distribution of false positives and false negatives is desired.
*   **Precision:** Focuses on the accuracy of positive predictions, minimizing false positives.
*   **Recall:** Focuses on the model's ability to find all positive cases, minimizing false negatives.

## Performance Metrics

The evaluation of each model under these strategies was quantified using the following metrics:

*   **Accuracy:** The proportion of correctly classified instances (both true positives and true negatives) out of the total instances.
*   **Precision Score:** The ratio of true positive predictions to the total positive predictions (true positives + false positives).
*   **Recall Score:** The ratio of true positive predictions to the total actual positive instances (true positives + false negatives).
*   **F1 Score:** The harmonic mean of Precision and Recall, providing a single metric that balances both.
*   **APS (Average Precision Score):** The area under the precision-recall curve, useful for imbalanced datasets.
*   **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** A measure of a classifier's performance across all possible classification thresholds, indicating its ability to distinguish between classes.

## Results
Here is the original data in a markdown table format:



| Model           | Evaluation Strategy | Accuracy | Precision Score | Recall Score | F1 Score | APS Score | ROC-AUC |
| :-------------: | :-----------------: | :------: | :-------------: | :----------: | :------: | :-------: | :-----: |
| LGBM            | F1                  | 0.999    | 0.8493          | 0.7881       | 0.8176   | 0.6697    | 0.894   |
|                 | Precision           | 0.9995   | 0.8981          | 0.7839       | 0.8371   | 0.7043    | 0.8919  |
|                 | Recall              | 0.9764   | 0.8298          | 0.8855       | 0.8444   |**0.9527** |**0.9377**|
| XGBoost         | F1                  | 0.9993   | 0.8122          | 0.7881       | 0.8      | 0.6405    | 0.8939  |
|                 | Precision           |**0.9999**| 0.8202          | 0.7924       | 0.806    | 0.6602    | 0.896   |
|                 | Recall              | 0.9804   | 0.0695          | 0.8686       | 0.1286   | 0.0606    | 0.9246  |
| CatBoost (CB)   | F1 score            | 0.9995   | 0.9474          | 0.7687       | 0.845    | 0.733     | 0.8813  |
|                 | Precision           | 0.9995   | 0.9524          | 0.7627       | 0.7968   | 0.7043    | 0.8813  |
|                 | Recall              | 0.9995   | **0.9596**      | 0.7869       |**0.8498**| 0.731     | 0.8834  |
| Isolation Trees | F1 Score            | 0.9972   | 0.2524          | 0.3347       | 0.2878   | 0.085     | 0.6665  |
|                 | Precision           | 0.9982   | 0.375           | 0.1144       | 0.1753   | 0.0444    | 0.557   |
|                 | Recall              | 0.5099   | 0.0033          | **0.9704**   | 0.0065   | 0.0238    | 0.7397  |




## Key Findings and Trends

The analysis of the model evaluation data revealed several interesting trends and patterns:

### 1. Overall High Performance for Gradient Boosting Models (LGBM, XGBoost, and CatBoost)

*   **Consistent Strong Performance:** LGBM, XGBoost, and CatBoost (CB) consistently achieved high scores across Accuracy, Precision, Recall, F1, APS, and ROC-AUC metrics. This indicates their robustness and effectiveness for the underlying classification task.
*   **Balanced Precision and Recall for LGBM and CatBoost (CB):** These models generally maintained a good balance between Precision and Recall across different evaluation strategies, leading to high F1 scores. This suggests they are effective at both identifying positive cases correctly and minimizing missed positive cases.
*   **High ROC-AUC Scores:** LGBM, XGBoost, and CatBoost consistently showed high ROC-AUC scores (typically above 0.88), signifying their strong ability to discriminate between positive and negative classes.

### 2. Isolation Trees Show Distinctly Lower Performance

*   **Significantly Lower Metrics:** Isolation Trees exhibited substantially lower performance across all metrics compared to the other models. This is particularly evident in the F1 Score, APS Score, and ROC-AUC, suggesting it is not well-suited for this specific classification task.
*   **Extreme Trade-off in "Recall" Strategy:** When evaluated under the "Recall" strategy, Isolation Trees achieved a very high Recall Score (0.9704). However, this came at the expense of an extremely low Precision Score (0.0033) and F1 Score (0.0065). This indicates that while it could identify nearly all positive cases, it also produced a massive number of false positives, making its predictions largely unreliable for practical use where precision is important.

### 3. Impact of Evaluation Strategy on Model Performance

*   **Minimal Impact for LGBM and CatBoost (CB):** For LGBM and CatBoost (CB), the choice of evaluation strategy (F1, Precision, or Recall) did not lead to drastic changes in their overall performance metrics. The scores remained relatively high and consistent, indicating their inherent stability.
*   **XGBoost Sensitivity to "Recall" Strategy:** While generally a strong performer, XGBoost showed a notable drop in Precision Score (0.0695) and F1 Score (0.1286) when explicitly optimized for "Recall." Despite maintaining a high Recall Score (0.8686) in this scenario, the significant reduction in Precision highlights a clear trade-off. This suggests that optimizing XGBoost solely for recall can compromise its ability to make accurate positive predictions.

## Conclusion

Based on this evaluation:

*   **LGBM, XGBoost, and CatBoost (CB) are highly recommended** for this classification task due to their consistently strong and balanced performance across various metrics and evaluation strategies.
*   **LGBM and CatBoost (CB)** demonstrated slightly more balanced performance, especially when considering the trade-offs between precision and recall.
*   **XGBoost** is also a strong candidate, but care must be taken if the goal is solely to maximize recall, as this can lead to a significant drop in precision.
*   **Isolation Trees are not suitable** for this task given their significantly lower performance and their tendency to produce highly imprecise results when recall is maximized.

Further investigation into specific business requirements and tolerance for false positives/negatives would help in selecting the absolute best model and evaluation strategy for deployment.

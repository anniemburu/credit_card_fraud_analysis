

``` markdown
# Credit Card Analysis : Supervised vs Unsupervised

## Overview

This document presents a detailed analysis of the performance of several machine learning models under different evaluation strategies. The primary goal is to identify the most effective models for a given task (implied to be a classification task based on the metrics) and understand how various evaluation approaches influence their reported performance.

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

```

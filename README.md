# Bayesian and Regularized Logistic Regression Analysis

## Objective

This project aims to model and analyze the relationship between predictors: **Age**, **Income**, and **Duration_Months** and a binary outcome variable (**Success**). The goal is to estimate the influence of these variables on the probability of success, assess model fit, interpret coefficient effects, and compare Bayesian logistic regression results with frequentist regularized logistic regression (L1 and L2 penalties). 

---

## Tools and Technologies Used

- **Python**: Primary programming language.
- **PyMC (v5.x)**: For Bayesian logistic regression modeling.
- **ArviZ**: For Bayesian model diagnostics and visualization.
- **scikit-learn**: For regularized logistic regression (L1 and L2) and performance evaluation.
- **NumPy & Pandas**: Data manipulation and preparation.
- **Matplotlib & Seaborn**: Visualization of results and diagnostic plots.

---

## Data Description

The dataset consists of several features and a binary outcome variable:

| Feature          | Type       | Description                      |
|------------------|------------|--------------------------------|
| Age              | Continuous | Age of the individual           |
| Income           | Continuous | Income of the individual        |
| Duration_Months  | Continuous | Duration of a specific condition or membership in months |
| Success          | Binary     | Outcome variable (0 = failure, 1 = success) |

---

## Modeling Approach

### 1. Bayesian Logistic Regression

- Built a Bayesian logistic regression model using **PyMC**.
- Initially modeled with raw inputs, then with **standardized predictors** (mean=0, std=1) to enable direct coefficient comparison.
- Posterior distributions of coefficients and intercept were estimated using MCMC sampling.
- Model convergence diagnostics included:
  - Traceplots (visual check for chain mixing)
  - Effective Sample Size (ESS)
  - R-hat statistics
- Model fit assessed by Bayesian R² and posterior predictive checks (limited by software version issues).
- Calculated **odds ratios** from posterior means to interpret effect size in odds terms.

### 2. Regularized Logistic Regression

- Employed **L1 (Lasso)** and **L2 (Ridge)** penalized logistic regression models using **scikit-learn**.
- Input features were standardized for consistency.
- Models evaluated on classification metrics:  
  - Accuracy  
  - Precision, Recall, F1-score  
  - ROC AUC  
- Coefficients from both L1 and L2 models were compared against Bayesian posterior means.

---

## Results and Interpretation

### Bayesian Model (Standardized Inputs)

| Predictor       | Mean Coefficient | 95% Credible Interval | Mean Odds Ratio | OR 95% Credible Interval  |
|-----------------|------------------|----------------------|-----------------|---------------------------|
| Age             | 0.669            | [0.52, 0.83]         | 1.95            | [1.68, 2.28]              |
| Income          | 0.378            | [0.23, 0.52]         | 1.46            | [1.26, 1.69]              |
| Duration_Months | 0.758            | [0.60, 0.92]         | 2.13            | [1.83, 2.50]              |
| Intercept       | 1.047            | [0.89, 1.20]         | 2.85            | [2.44, 3.32]              |

- All predictors had credible intervals excluding zero, indicating strong positive association with success.
- Duration_Months had the largest effect on odds of success, followed by Age and Income.
- Model convergence diagnostics (ESS, R-hat) indicated stable and reliable posterior samples.
- Bayesian R² showed moderate explanatory power (~0.19).

### Regularized Logistic Regression

- Both L1 and L2 models achieved similar classification performance:
  - Accuracy: ~76.5%
  - ROC AUC: ~0.735
- Coefficients for standardized predictors were close to Bayesian estimates:
  - Age ~ 0.63
  - Income ~ 0.39
  - Duration_Months ~ 0.79
- The alignment of Bayesian and frequentist results strengthens confidence in the findings.

---

## Model Validation and Diagnostics

- **Bayesian Diagnostics**: Traceplots, ESS, R-hat statistics confirmed good MCMC convergence.
- **Posterior Predictive Checks (PPC)** were attempted but limited due to library version compatibility issues.
- **Bayesian R²** was used as a model fit metric.
- **Frequentist Metrics**: Accuracy, Precision, Recall, F1-score, ROC AUC measured classification performance for regularized logistic regression.

---

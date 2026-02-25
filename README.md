# Credit Scoring Evaluation Using Machine Learning

**Diploma Thesis - National Technical University of Athens (NTUA)**  
School of Applied Mathematical and Physical Sciences, Department of Mathematics  
September 2022

**Author:** Emmanouil Fosteris  
**Supervisor:** Prof. Chryseis Karoni

---

## Abstract

This thesis studies credit scoring using machine learning techniques. The aim is to provide a detailed analysis of various modeling techniques, the data used, and an evaluation of their predictive performance.

The study applies several supervised learning methods to a credit scoring problem with a sample of **366 borrowers**, predicting the probability of **loan default within a five-year period**. The models evaluated include:

- **Logistic Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Elastic Net Regression**
- **Decision Trees**
- **Random Forest**

The results demonstrate that all techniques were able to explain the data to a satisfactory degree. Logistic regression slightly outperformed the other methods with an **AUC of 0.7458** on the test set, though all models produced reliable results suitable for use by financial institutions.

## Repository Structure

```
├── README.md
├── code/
│   └── credit_scoring_analysis.R    # Full R analysis code
├── docs/
│   └── Emmanouil_Fosteris_Thesis_Final.pdf  # Full thesis document (in Greek)
└── LICENSE
```

## Key Features & Variables

The dataset consists of 366 observations split into a training set (302) and test set (64). The binary target variable is **loan default within 5 years** (Yes/No).

**Predictor Variables:**

| Variable | Description | Categories |
|----------|-------------|------------|
| Solvency | Settled adverse items | 0: Above €1,500 · 1: Below €1,500 |
| Property | Assets-to-debt ratio | 0: Up to 100% · 1: 100%–300% · 2: Above 300% |
| History | Credit history | 0: No previous loans · 1: Existing loan (no issues) · 2: Previous loan fully repaid |
| Employment | Employment status | 0: Private sector · 1: Public sector · 2: Professional (e.g., doctor) · 3: Other self-employed |

## Results Summary

| Model | AUC (Test Set) | 95% CI |
|-------|---------------|--------|
| Logistic Regression | **0.7458** | [0.5889, 0.8791] |
| Ridge Regression | 0.7453 | [0.5825, 0.8799] |
| Decision Tree | 0.7259 | [0.5554, 0.8594] |
| Random Forest | 0.7205 | [0.5678, 0.8573] |
| Lasso Regression | 0.7054 | [0.5478, 0.8475] |

## Key Findings

- **Employment status** was the most significant predictor of loan default across all models
- **Credit history** was found to be statistically insignificant and was removed from the final models
- Public sector employees showed significantly lower default probability compared to other employment categories
- Logistic regression provided the best balance of interpretability and predictive performance

## Technologies

- **Language:** R
- **Key Libraries:** `glmnet`, `pROC`, `rpart`, `randomForest`, `ggplot`, `dplyr`

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/manolisfosteris/credit-scoring-thesis.git
   ```
2. Open `code/credit_scoring_analysis.R` in RStudio
3. Install required packages:
   ```r
   install.packages(c("dplyr", "pROC", "glmnet", "ggplot2", "rpart",
                       "cowplot", "randomForest", "tree", "xtable",
                       "glmtoolbox", "lmtest", "rpart.plot"))
   ```
4. Run the script (note: the original dataset is not publicly included due to privacy considerations)

## License

This work is licensed for non-commercial, educational, and research purposes. See [LICENSE](LICENSE) for details.




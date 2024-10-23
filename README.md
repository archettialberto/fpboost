# FPBoost: Fully Parametric Gradient Boosting for Survival Analysis

![License Banner](https://img.shields.io/badge/License-MIT-purple.svg)
![Black Banner](https://img.shields.io/badge/code%20style-black-000000.svg)

FPBoost is a Python library for survival analysis that introduces a novel algorithm for estimating
hazard functions. Built upon the gradient boosting framework, it uses a composition of fully
parametric hazard functions to model time-to-event data. FPBoost directly optimizes the survival
likelihood via gradient boosting, providing improved risk estimation according to concordance and
calibration metrics. FPBoost is fully compatible with
[scikit-survival](https://scikit-survival.readthedocs.io/en/stable/index.html) for seamless
integration into existing workflows.

## 📦 Installation

### From Source

To install the latest version of FPBoost from source, clone the repository and follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/archettialberto/fpboost.git
    cd fpboost
    ```
2. Create and Activate Conda Environment
    ```bash
    conda env create -f environment.yaml
    conda activate fpboost
    ```
3. Install Dependencies with Poetry
    ```bash
    poetry install
    ```

## 🚀 Quick Start
Here's a simple example of how to use FPBoost:

```python
from fpboost.models import FPBoost
from sksurv.datasets import load_breast_cancer
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
data_x, data_y = load_breast_cancer()
encoder = OneHotEncoder()
X, y = encoder.fit_transform(data_x).to_numpy(), data_y

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the model
model = FPBoost(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Fit the model
model.fit(X_train, y_train)

# Predict survival probabilities
surv_probs = model.predict_survival_function(X_test)

# Evaluate the model
from sksurv.metrics import concordance_index_censored
c_index = concordance_index_censored(
    y_test['e.tdm'],  # event indicator
    y_test['t.tdm'],  # time to event
    model.predict(X_test)
)

print("Concordance Index:", c_index[0])
```

## 📖 Documentation

For detailed usage instructions and API reference, please refer to the [FPBoost Documentation]().

## 📚 How to Cite

If you use FPBoost in your research, please cite our paper:

```bibtex
@article{archetti2024fpboost,
  title        = {FPBoost: Fully Parametric Gradient Boosting for Survival Analysis},
  author       = {Alberto Archetti and Eugenio Lomurno and Diego Piccinotti and Matteo Matteucci},
  journal      = {arXiv preprint arXiv:2409.13363},
  year         = {2024},
  url          = {https://arxiv.org/abs/2409.13363}
}
```

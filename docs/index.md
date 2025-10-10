# AutoRA Experimentalist Challenge

This project benchmarks a **custom experimentalist strategy** against **random sampling** in a closed-loop scientific discovery process using [AutoRA](https://autoresearch.github.io/autora/).

We simulate experiments in two ground-truth domains:
- Task Switching (Neuroscience)
- Expected Value Theory (Economics)

---

## 🧪 Overview: Closed-Loop Discovery

Each experiment loop consists of:
1. **Training theorists** on collected data
2. **Selecting new experimental conditions** via an experimentalist
3. **Running the experiment** (synthetic environment)
4. **Evaluating performance** on a fixed validation set

---

#  Experimentalist Documentation

##  Inputs

Our custom experimentalist, `sample_flavour`, takes two key inputs:

- **conditions**:  
  A set of previously sampled experiment points (shape: `[n_samples, n_features]`), representing known data.
  
- **candidates**:  
  A pool of potential new experiment points (shape: `[n_candidates, n_features]`), typically generated via grid or random sampling.

These inputs allow the experimentalist to make informed decisions by comparing **known data** (exploration history) against **available options** (future actions).

We consider these inputs to:

- Focus sampling around **regions where data already exists** (exploitation)
- Maintain potential for **broader coverage** (exploration), controlled via parameters

---

## 🧪 Sampling Method

Our experimentalist uses a **proximity-weighted sampling method**, specifically:

### 🌀 Gaussian Proximity Kernel

For each candidate, we compute its proximity to known conditions using:

$$
\text{proximity}(x) = \sum_{x_i \in \text{conditions}} \exp\left(-\frac{||x - x_i||^2}{2\sigma^2}\right)
$$

This creates a **soft similarity score**: candidates closer to known points get higher scores.

We then apply a **softmax reweighting** (with temperature scaling):

$$
P(x) = \frac{\exp(\text{proximity}(x) / T)}{\sum_{x'} \exp(\text{proximity}(x') / T)}
$$

Where:
- $\sigma$ controls the **width** of the proximity kernel
- $T$ (temperature) controls **exploration-exploitation**:
  - Low $T$ → favors closest candidates (exploitation)
  - High $T$ → more uniform selection (exploration)

### ✅ Why this method?

We aim to:
- Encourage **local refinement** (near known data)
- Avoid getting stuck in a small region by controlling the **exploration level**
- Provide a **simple but principled heuristic** without needing full uncertainty quantification or model disagreement

This makes `sample_flavour` a lightweight and interpretable alternative to more complex acquisition functions.

---

## 📊 Experimentalists Compared

| Experimentalist       | Strategy                      |
|------------------------|-------------------------------|
| Random Sampling        | Uniform random draw           |
| `sample_flavour`       | Proximity-weighted softmax    |

---

## 📉 Evaluation

We compare experimentalists using **validation Mean Squared Error (MSE)** from theorists (BMS, Linear, Polynomial) trained on data collected across discovery cycles.

Lower MSE = Better scientific model discovery.

---

## 🧠 Theorists

- BMS Regressor (Bayesian)
- Linear Regression
- Polynomial Regression (degree = 3)

---

## 📂 File Structure

```text
autora-experimentalist-challenge/
│
├── src/
│   └── autora_experimentalist_example/
        └──__init__.py  # sample_flavour implementation
        └── proximity_functions.py 
        └── test_proximity_functions.py
│
├── notebooks/
│   └── benchmark_experimentalists.ipynb   # run & visualize experiment
│
├── doc/
│   └── index.md 
Basic                          # [this documentation]
├── requirements.txt
└── ...

# EVRP Optimization with Dynamic Traffic Flow

This repository contains the implementation for my undergraduate thesis: **"Research on Electric Vehicle Routing Problem (EVRP) Considering Traffic Flow Influence"**.

The project proposes a hybrid framework integrating **Deep Learning** (CEEMDAN-LSTM) and **Heuristic Optimization** (Improved Genetic Algorithm) to solve logistics routing problems under dynamic traffic constraints.

## 📂 Repository Structure

```text
├── CEEMDAN_LSTM/               # Traffic Prediction Module
│   ├── data/                   # Historical traffic datasets (Guangzhou, 2016)
│   ├── models/                 # PyTorch implementation of LSTM
│   └── ceemdan_process.py      # CEEMDAN signal decomposition
│
├── MY_GA/                      # Optimization Module
│   ├── datasets/               # Modified Solomon benchmarks
│   ├── ga_solver.py            # GA with Elite Preservation & Heuristic Initialization
│   └── cost_functions.py       # Speed-dependent energy models
│
├── undergraduate_thesis.pdf    # Full Thesis (Chinese)
└── README.md
```
## 🧠 Core Methodology

### 1. Traffic Prediction (CEEMDAN-LSTM)
* **Decomposition**: Utilized **CEEMDAN** to decompose non-stationary traffic velocity signals into Intrinsic Mode Functions (IMFs).
* **Prediction**: Implemented **LSTM** networks to predict the trend of each component, reconstructing dynamic travel speeds for the routing model.

### 2. Route Optimization (Improved GA)
* **Dynamic Modeling**: Incorporates **speed-dependent energy consumption** instead of constant energy assumptions.
* **Algorithm**: Enhanced Genetic Algorithm with **Elite Preservation** strategies to prevent solution degradation and **Heuristic Initialization** (Nearest Neighbor) to accelerate convergence.

## 📊 Key Findings

* **Reality Gap**: Validated that static models underestimate logistics costs by **20.4%** compared to this dynamic traffic model.
* **Strategy Analysis**:
    * **Battery Swapping**: Most cost-effective when the unit swapping cost is less than **10x** the charging cost.
    * **Partial Charging**: The "80% Partial Charging" strategy reduces total costs by **8.4%** in high time-penalty scenarios.


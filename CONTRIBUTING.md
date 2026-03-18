# Contributing Guidelines

Thank you for your interest in contributing to **dice-x (DiCE-Extended)**.

This repository is part of an ongoing research project. Therefore, contributions are subject to strict guidelines to ensure consistency, reproducibility, and scientific validity.

---

## How to Contribute

All contributions must follow this workflow:

1. Fork the repository  
2. Create a new branch  
3. Implement your changes  
4. Submit a **Pull Request (PR)**  

---

## Pull Request Requirements

Every Pull Request **must include**:

- A clear and detailed explanation of the changes  
- Motivation for the change (why it is needed)  
- Description of affected components  
- **Experimental or test results** demonstrating the impact of the change  

PRs without proper explanation or results **will not be reviewed**.

---

## Allowed Contributions

The following types of contributions are welcome:

### 1. Performance Improvements
- Runtime optimisations  
- Memory efficiency improvements  
- Better batching or vectorisation  
- Hardware-aware optimisations (e.g., GPU usage)

---

### 2. Engineering Improvements
- Code refactoring (without changing logic)  
- Improved modularity or readability  
- Logging, monitoring, or pipeline improvements  

---

### 3. Bug Fixes
- Fixing implementation errors  
- Resolving incorrect computations  
- Addressing numerical instability  

---

### 4. Mathematical Corrections
- Identifying and correcting incorrect implementations of existing methods  
- Providing justification based on literature or derivations  

---

## Not Allowed Contributions

The following are **strictly prohibited**:

### Modifying Core Algorithms
- Changes to the **core counterfactual generation logic**  
- Introducing new optimisation objectives that alter method behaviour  
- Altering loss functions or search strategies  

---

### New Methods or Research Ideas
- Adding new counterfactual generation techniques  
- Proposing alternative frameworks  

> This repository focuses on evaluating and stabilising existing methods, not introducing new ones.

---

### Unverified Claims
- Claims of improvement without experimental evidence  
- Subjective performance claims  

---

## Experimental Requirements

If your contribution affects performance, you must include:

- Dataset used  
- Model used  
- Metrics (e.g., validity, proximity, diversity, robustness)  
- Before vs after comparison  

---

## Code Style

- Follow existing project structure  
- Write clear, modular, and readable code  
- Add comments where necessary  
- Avoid unnecessary dependencies  

---

## Review Process

1. Initial screening for guideline compliance  
2. Technical and experimental evaluation  
3. Decision: **accepted / revision requested / rejected**

---

## Important Note

This repository is part of an academic research effort.  
Maintaining **scientific correctness and reproducibility** is the highest priority.

Contributions that compromise these principles will be rejected.

---

## Final Words

We appreciate your effort and interest in improving the project.  
Please ensure your contribution aligns with the goals of **robust, reliable, and reproducible counterfactual evaluation**.
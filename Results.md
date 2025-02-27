# Introduction to the extended version of DiCE (Diverse Counterfactual Explanations)

[Mothilal et al. (2020)](https://dl.acm.org/doi/10.1145/3351095.3372850) introduce their method of generating counterfactual explanations considering _feasibility_, and _diversity_. [Guidotti and Ruggieri (2021)](https://link.springer.com/chapter/10.1007/978-3-030-88942-5_28), claim counterfactual explanations to be robust they should be similar for similar instances when they explain. In this study, in a search to improve the quality and reliability of the counterfactual explanations _robustness_ is found to be helpful and it also introduced in the optimization function.

DiCE-Extended is built upon the [DiCE (Diverse Counterfactual Explanations)](https://github.com/interpretml/DiCE) [(Mothilal et al. 2020)](https://dl.acm.org/doi/10.1145/3351095.3372850) framework by introducing a robustness term in the optimization function.

## Manipulated Optimization Function

The core enhancement in DiCE-Extended is the manipulated optimization function, designed to balance proximity, diversity, and feasibility of counterfactuals. The function is formulated as:

$$
C(x) = \underset{c_1, ..., c_k}{\text{arg min}}
\frac{1}{2} \sum_{i=1}^{k} yloss(f(c_i), y) +
\frac{\lambda_1}{k} \sum_{i=1}^{k} dist(c_i, x) -
\lambda_2 \cdot dpp\_diversity(c_1, ..., c_k) -
\frac{\lambda_3}{k} \sum_{i=1}^{k} robustness(c_i, c_i')
$$

- **Proximity Loss**: The first term that averages the distance between generated counterfactuals and the original input ensure the counterfactuals to be as close as possible to the original input.
- **Diversity Loss**: Diversity of the counterfactual explanations is aquired by determinental point process of which loss is represented by the second term and it ensures that _k_ number of counterfactual explanations are generated.
- **Robustness Loss**: [Guidotti (2024)](https://link.springer.com/article/10.1007/s10618-022-00831-6) defines robustness as necessity of similar instances being explained by similar counterfactual explanations such that if $b(x_1)=b(x_2)=y$ then an explainer $f$ should generate counterfactuals $c_1$ and $c_2$ that are similar and can explain $x_1$ and $x_2$. The robustness term that is based on [Dice-Sørensen Coefficient](https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient), is adopted from [Bonasera and Carrizosa (2024)](
https://doi.org/10.48550/arXiv.2407.00843).

$$
Robustness(c_i, c_i') = \frac{2 * \lvert c_i \cap c_i' \rvert}{\lvert c_i \rvert + \lvert c_i' \rvert}
$$


By adjusting the weights $\lambda_1$, $\lambda_2$, $\lambda_3$ counterfactual explanations can be customised by specific needs.

### IMPORTANT NOTE

Some of the calculations in this notebook may yield slightly different results across runs. This variability is due to the stochastic nature of optimization processes, random initializations, and other computation dependent factors. Please keep this in mind when interpreting the results. For reproducibility, consider setting random seeds where applicable.

### Explainers' Loss Chart

During the counterfactual generation process we keep track of each losses computed for the optimization process. The chart below shows that the counterfactuals are optimized depending on various type of losses i.e., class loss, proximity loss, diversity loss, and robustness loss. The algorithm although reached a plateau around 50th iteration since the stopping criteria which is loss difference isn't met the loop keeps the calculations. The other stopping criteria is maximum number of iterations which is set to 5000. The chart is plotted without considering the weights for losses. 

![Explainer History](result_images/explainer_history.png)

### Metrics Chart

We obtained different metrics that are diversity, proximity, sparsity specified in [Mothilal et al. (2020)](https://dl.acm.org/doi/10.1145/3351095.3372850). Additionally, we added robustness as the new metric. The chart below shows the results of these metrics with different datasets and counterfactual generation strategies.

![Metric Chart](result_images/metrics_chart.png)

### Results on counterfactual datasets

We trained 12 models for each four dataset and 3 counterfactual generation strategies. The table below shows the accuracy, f1 score, recall, precision, and auc scores on the test dataset.

| dataset           | backend | accuracy | f1_score | recall | precision | auc  |
|-------------------|---------|----------|----------|--------|-----------|------|
| adult-income      | sklearn | 0.98     | 0.99     | 0.99   | 0.98      | 0.98 |
| adult-income      | PYT     | 0.99     | 0.99     | 1.00   | 1.00      | 1.00 |
| adult-income      | TF2     | 1.00     | 1.00     | 1.00   | 0.99      | 1.00 |
| compas-recidivism | sklearn | 1.00     | 1.00     | 1.00   | 1.00      | 1.00 |
| compas-recidivism | PYT     | 1.00     | 1.00     | 1.00   | 1.00      | 1.00 |
| compas-recidivism | TF2     | 1.00     | 1.00     | 1.00   | 1.00      | 1.00 |
| german-credit     | sklearn | 0.97     | 0.97     | 0.97   | 0.97      | 0.97 |
| german-credit     | PYT     | 1.00     | 1.00     | 1.00   | 1.00      | 1.00 |
| german-credit     | TF2     | 1.00     | 1.00     | 1.00   | 1.00      | 1.00 |
| lending-club      | sklearn | 0.92     | 0.92     | 0.87   | 0.98      | 0.92 |
| lending-club      | PYT     | 0.99     | 0.99     | 1.00   | 1.00      | 1.00 |
| lending-club      | TF2     | 0.99     | 0.99     | 0.98   | 1.00      | 0.99 |

The counterfactual datasets generated with 4 datasets and 3 different counterfactual generation strategies is investigated in terms of validity, diversity, and robustness.

The validity of each dataset for different backend algorithms are as below:

| VALIDITY SCORE    | sklearn | PYT  | TF2  |
|-------------------|---------|------|------|
| adult-income      | 0.89    | 0.97 | 0.99 |
| lending-club      | 0.77    | 0.76 | 0.87 |
| german-credit     | 0.45    | 0.48 | 0.99 |
| compas-recidivism | 0.66    | 0.58 | 0.90 |

The continuous diversity of each dataset for different backend algorithms are as below:

| Continuous Diversity | sklearn | PYT  | TF2  |
|----------------------|---------|------|------|
| adult_income         | 2.35    | 1.62 | 1.68 |
| lending-club         | 1.83    | 2.31 | 5.39 |
| german-credit        | 1.21    | 1.42 | 1.37 |
| compas-recidivism    | 2.97    | 2.51 | 1.89 |

The categorical diversity of each dataset for different backend algorithms are as below:

| Categorical Diversity | sklearn | PYT  | TF2  |
|-----------------------|---------|------|------|
| adult_income          | 0.50    | 0.64 | 0.66 |
| lending-club          | 0.75    | 0.62 | 0.60 |
| german-credit         | 0.51    | 0.58 | 0.58 |
| compas-recidivism     | 0.43    | 0.49 | 0.48 |

To ensure the consistency for the desired classes and randomization we generated 5 counterfactuals until we reached the desired number for each classes. Unfortunately, because of lacking the original instances used for counterfactual generation it wasn't possible to compute an overall proximity score.

| Robustness Score   | sklearn | PYT  | TF2  |
|--------------------|---------|------|------|
| adult-income       | 1.00    | 0.39 | 0.53 |
| lending-club       | 0.98    | 0.49 | 0.40 |
| german-credit      | 0.99    | 0.61 | 0.53 |
| compas-recidivism  | 1.00    | 0.29 | 0.27 |

### Visually comparing (PCA) the original dataset and counterfactual dataset

Here we will show how the original datasets and counterfactual datasets lie on the plot to analyze whether the counterfactuals are within the original datasets boundaries.

![PCA Chart](result_images/PCA_chart.png)

## Models' accuracies on all four datasets

We trained two types of model with three different frameworks. Firstly, a tree model is created by using sci-kit learn framework's RandomForestClassifier with the datasets with default settings. The other two models are created with PyTorch and TensorFlow frameworks. These neural networks have two layers and the architecture is same as present in the DiCE repository as follows:

$$
Linear(number\_of\_features, 20) → ReLU → Linear(20, 1) → Sigmoid
$$

While training the neural networks following hyperparameters are used:
- Learning rate: 0.001
- Number of epochs: 10
- Train dataset size: Dataset size * 80%
- Test dataset size: Dataset size * 20%
- Number of batches for training dataset: 16
- Number of batches for test dataset: 4
- Optimizer: Adam

| Dataset           | Model                    | Accuracy   |
|:------------------|:-------------------------|:-----------|
| compas-recidivism | Random Forest Classifier | 57.14%     |
|                   | Neural Network (PYT)     | 65.79%     |
|                   | Neural Network (TF2)     | 65.49%     |
| adult-income      | Random Forest Classifier | 81.84%     |
|                   | Neural Network (PYT)     | 83.36%     |
|                   | Neural Network (TF2)     | 83.43%     |
| lending-club      | Random Forest Classifier | 82.39%     |
|                   | Neural Network (PYT)     | 82.92%     |
|                   | Neural Network (TF2)     | 82.94%     |
| german-credit     | Random Forest Classifier | 75.5%      |
|                   | Neural Network (PYT)     | 77.5%      |
|                   | Neural Network (TF2)     | 78.5%      |


As it can be seen from the table the models performed satisfying both with [Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult) and the [Lending Club Dataset](https://www.lendingclub.com/). While models perform moderately on [German Credit Risk Dataset](https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip), they perform poorly on [Compas Recidivism Dataset](https://api.openml.org/data/download/22111929/dataset).


## Explainers' counterfactual generation time with different datasets

Three types of explainers are generated for counterfactual generation that are genetic, PyTorch, and TensorFlow. Counterfactuals generated by genetic algorithm are generated with a genetic algorithm creates mutations with the best counterfactuals depending on the loss value. The gradient methods use Adam optimizer with a learning rate of $0.05$ and parameterize the counterfactuals to optimize counterfactuals. During the counterfactual generation process 5 counterfactuals has been created. The table below shows time spent for counterfactual generation for each model and dataset. PYT represents a neural network model created with PyTorch and TF2 represents a neural network model created with TensorFlow framework.

| Dataset           | Model          |   Time (s) |
|:------------------|:---------------|------------:|
| compas-recidivism | Genetic        |       10.22 |
| compas-recidivism | Gradient (PYT) |        2.69 |
| compas-recidivism | Gradient (TF2) |       30.95 |
| adult-income      | Genetic        |       49.08 |
| adult-income      | Gradient (PYT) |        7.14 |
| adult-income      | Gradient (TF2) |       43.72 |
| lending-club      | Genetic        |      110.42 |
| lending-club      | Gradient (PYT) |       17.32 |
| lending-club      | Gradient (TF2) |      200.41 |
| german-credit     | Genetic        |      217.62 |
| german-credit     | Gradient (PYT) |       10.89 |
| german-credit     | Gradient (TF2) |       61.37 |

## Metrics and Sensitivity Analysis for Dice Extended


### 1. Robustness Metrics

#### Dice-Sørensen Coefficient

To evaluate robustness, the Dice-Sørensen coefficient measures the similarity between counterfactuals c1 and
c2 generated for similar input instances x1 and x2:

$$
Robustness(c_1, c_2) = \frac{2 * \lvert c_1 \cap c_2 \rvert}{\lvert c_1 \rvert + \lvert c_2 \rvert}
$$

where:
- $ c_1 $ and $ c_2 $ are binary vectors,
- $ \lvert c_1 \cap c_2 \rvert $: The number of shared (overlapping) features between c1 and c2,
- $ \lvert c_1 \rvert $ and $ \lvert c_2 \rvert $: The total number of features in each counterfactual.

#### Input Perturbation and Stability

Stability under input perturbation measures the solution variance when slight perturbations are introduced
to the input instance. The procedure includes the following steps:

1) **Apply Gaussian Noise:** Perturb the input $x$ by adding Gaussian noise $\delta$ to create perturbed inputs
$x'$:

$$
x' = x + \delta, \quad \delta \sim \mathcal{N}(0, \sigma^2)
$$

where $\sigma$ is the standard deviation of the noise (e.g., $\sigma = 0.01$).

2) **Generate Counterfactuals:** Generate counterfactual explanations $c_i$ for the original input $x$ and $c_i'$ for the perturbed input $x'$.

3) **Measure Stability:** Compare counterfactuals using a distance metric, such as the Euclidean distance:

$$
Stability = \frac{1}{n} \sum_{i=1}^{n} dist(c_i, c_i')
$$

where:

$$
dist(c_i, c_i') = \sqrt{\sum_{j=1}^{d} (c_{ij} - c_{ij}')^2}
$$

$n$ is the total number of input instances, $c_i$ is the counterfactual for the original input, and $c_i'$ is the counterfactual for the perturbed input.

In the section below we will do the required computation to calculate the stability of the counterfactuals. Firstly we pick an instance from the dataset which is $x$ and generate counterfactuals to it. Subsequently, we perturb $x$ and obtain $x'$ and generate counterfactuals also for it.

#### Result for Stability

We computed the stability metric by converting the $C$ and $C'$ into a normalized vector that are of shape (10, 70) which means there are 10 samples with 70 features. The result of the calculation of stability under input perturbation is $~2.07$. When we consider the maximum distance between two vectors of shape (10, 70) maximum distance should equal to $~8.37$

$$
\sqrt{70 \times (1 - 0)} = 8.37
$$

### 2. Counterfactual Quality Measures

#### Fidelity

Fidelity measures how often generated counterfactuals successfully change the model’s prediction:

$$
Fidelity = \frac{\sum_{i=1}^{n} \mathbf{1}(f(c_i) = y_{desired})}{n}
$$

where:

- $f$: Prediction model,
- $c_i$: Counterfactual instance,
- $y_{desired}$: Target output class,
- $n$: Total number of counterfactuals.

Since all generated counterfactuals belong to the desired class $fidelity$ of the counterfactuals is $100\%$

#### Proximity

Proximity measures the average distance between counterfactuals $c_i$ and the original inputs $x_i$:

$$
Proximity = \frac{1}{n} \sum_{i=1}^{n} dist(x_i, c_i)
$$

The Manhattan distance can be used for simplicity:

$$
dist(x_i, c_i) = \sum_{j=1}^{d} \lvert x_{ij} - c_{ij} \rvert
$$

#### Result for Proximity

We computed the proximity of $x$ and $C$ we converted them into normalized and one hot encoded tensors of shape (10, 70). The resulting total proximity between the original instance and generated counterfactuals is $7.65$ which is highly acceptable when we consider the maximum Manhattan distance between these two tensors which is $70$.

#### Diversity

Diversity measures how dissimilar the counterfactuals $c_1, c_2, c_3,\ldots,c_k$ are among themselves:

$$
Diversity = \frac{1}{k(k-1)}\sum_{i_1}^{k}\sum_{j \neq i}^{} dist(c_i, c_j)
$$

where $k$ is the number of counterfactuals.

#### Result for Diversity

For the counterfactuals tensor we work with minimum diversity value is $0$ that indicates all counterfactuals are identical. For the condition that all counterfactuals are distinct that makes the diversity value maximum which is $~4.19$. The diversity value we calculated with the counterfactuals is $1.87$ which seems that counterfactuals have a moderate spread in the features space.

### 3. Sensitivity Analysis

#### Objective Function with Weights

The modified loss function in DiCE-Extended is defined as in the given with the optimization function where:

- $yloss(f(c_i), y)$: Prediction loss for counterfactual instance $c_i$ relative to the desired outcome $y$,
- $dist(c_i, x)$: Distance metric (e.g., Euclidean or Manhattan) between the counterfactual c_i and the original input $x$,
- $dpp\_diversity(c_1,\ldots,c_k)$: Diversity loss term based on Determinantal Point Process (DPP),
- $Robustness(c_i,c_i')$: Robustness loss measuring similarity of counterfactuals under perturbations.

  The weights $\lambda_1, \lambda_2, \lambda_3$ control the balance between proximity, diversity, and robustness, respectively.

#### Sensitivity Analysis

To perform sensitivity analysis:

1) Vary the weights $\lambda_1, \lambda_2, \lambda_3$ systematically while ensuring:

$$
\lambda_1 + \lambda_2 + \lambda_3 = 1 \text{ (for normalization)}.
$$

2) Track the changes in the following metrics:

$$
P(\lambda_1, \lambda_2, \lambda_3) = Proximity,
$$

$$
D(\lambda_1, \lambda_2, \lambda_3) = Diversity,
$$

$$
R(\lambda_1, \lambda_2, \lambda_3) = Robustness,
$$

3) Measure the relationship between these metrics and the weights.

#### Proximity vs. Proximity Weight Chart

![Proximity vs. Proximity Weight](result_images/proximity_vs_proximity_weight.png)

#### Diversity vs Diversity Weight Chart

![alt text](result_images/diversity_vs_diversity_weight.png)

#### Robustness vs. Robustness Weight

![alt text](result_images/robustness_vs_robustness_weight.png)

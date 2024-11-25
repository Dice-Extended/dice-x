
# DiCE-Extended: Ensemble Selection of Diverse Counterfactual Explanations Using Continuous Optimization

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/dice-ml)

## Overview

[Mothilal et al. (2020)](https://dl.acm.org/doi/10.1145/3351095.3372850) introduce their method of generating counterfactual explanations considering _feasibility_, and _diversity_. [Guidotti and Ruggieri (2021)](https://link.springer.com/chapter/10.1007/978-3-030-88942-5_28), claim counterfactual explanations to be robust they should be similar for similar instances when they explain. In this study, in a search to improve the quality and reliability of the counterfactual explanations _robustness_ is found to be helpful and it also introduced in the optimization function.



DiCE-Extended is built upon the [DiCE (Diverse Counterfactual Explanations)](https://github.com/interpretml/DiCE) [(Mothilal et al. 2020)](https://dl.acm.org/doi/10.1145/3351095.3372850) framework by introducing a robustness term in the optimization function.

## Installation

DiCE-Extended requires Python 3.6 or higher. Install it using pip:

```bash
pip install dice-extended
```

Alternatively, install via conda:

```bash
conda install -c conda-forge dice-extended
```

## Getting Started

The following code piece provides an idea on how to generate counterfactual explanations with dice-extended (The code will be updated):

```python
import dice_ml
from dice_ml_x.utils import helpers
from sklearn.model_selection import train_test_split

# Load dataset
dataset = helpers.load_adult_income_dataset()
target = dataset["income"]
train_dataset, test_dataset, _, _ = train_test_split(
    dataset, target, test_size=0.2, random_state=0, stratify=target
)

# Initialize Data and Model
d = dice_ml.Data(
    dataframe=train_dataset,
    continuous_features=['age', 'hours_per_week'],
    outcome_name='income'
)
m = dice_ml.Model(
    model_path=dice_ml.utils.helpers.get_adult_income_modelpath(),
    backend='TF2',
    func="ohe-min-max"
)

# Generate Counterfactual Explanations
exp = dice_ml.Dice(d, m)
query_instance = test_dataset.drop(columns="income")[0:1]
dice_exp = exp.generate_counterfactuals(
    query_instance, total_CFs=4, desired_class="opposite"
)
dice_exp.visualize_as_dataframe()
```

## Manipulated Optimization Function

The core enhancement in DiCE-Extended is the manipulated optimization function, designed to balance proximity, diversity, and feasibility of counterfactuals. The function is formulated as:

$$ C(x) = \operatorname*{arg\,min}_{c_1, ... , c_k} \frac{1}{2} \sum_{i}^{k}yloss(f (c_i ), y) + \frac{\lambda_1}{k}\sum_{i}^{k}dist(c_i , x) - \lambda_2*dpp\_diversity(c_1, ... ,c_k) - \frac{\lambda_3}{k}\sum_{i}^{k}robustness(c_i, c_i')$$

- **Proximity Loss**: The first term that averages the distance between generated counterfactuals and the original input ensure the counterfactuals to be as close as possible to the original input.
- **Diversity Loss**: Diversity of the counterfactual explanations is aquired by determinental point process of which loss is represented by the second term and it ensures that _k_ number of counterfactual explanations are generated.
- **Robustness Loss**: [Guidotti (2024)](https://link.springer.com/article/10.1007/s10618-022-00831-6) defines robustness as necessity of similar instances being explained by similar counterfactual explanations such that if $b(x_1)=b(x_2)=y$ then an explainer $f$ should generate counterfactuals $c_1$ and $c_2$ that are similar and can explain $x_1$ and $x_2$. The robustness term that is based on [Dice-Sørensen Coefficient](https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient), is adopted from [Bonasera and Carrizosa (2024)](
https://doi.org/10.48550/arXiv.2407.00843).

By adjusting the weights $\lambda_1$, $\lambda_2$, $\lambda_3$ counterfactual explanations can be customised by specific needs.

## Evaluation

This part will be written at the end of the study.

## Contributing

We welcome contributions to DiCE-Extended. Please refer to our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citing

In case you may found useful this work for your research cite the original dice paper and this study's paper as well.

```
@inproceedings{mothilal2020dice,
  title={Explaining machine learning classifiers through diverse counterfactual explanations},
  author={Mothilal, Ramaravind K and Sharma, Amit and Tan, Chenhao},
  booktitle={Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency},
  pages={607--617},
  year={2020}
}

@article{sureyya2025dicex,
  title={Ensemble Selection of Diverse Counterfactual Explanations Using Continuous Optimization},
  author={Akyuz, Sureyya and Goktas, Polat and Bakir, Volkan},
  journal={Conference Proceedings (NeurIPS 2025)},
  year={2025}
}
```

## Acknowledgments

We extend our gratitude to the authors of [DiCE](https://github.com/interpretml/DiCE) for their foundational work in counterfactual explanations generation.

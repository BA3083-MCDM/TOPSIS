# TOPSIS

This repository contains a simple Python implementation of **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution) for multi-criteria decision making.

The main file is [TOPSIS.py](models/TOPSIS.py), which provides a `TOPSIS` class you can use to rank alternatives based on multiple criteria. A notebook version is also available in [TOPSIS.ipynb](examples/TOPSIS.ipynb).

## What TOPSIS Does

TOPSIS ranks alternatives by comparing each one to:

- the **positive ideal solution**: the best possible performance on each criterion
- the **negative ideal solution**: the worst possible performance on each criterion

An alternative is better if it is closer to the positive ideal and farther from the negative ideal.

## Files

- [TOPSIS.py](models/TOPSIS.py): reusable Python class
- [TOPSIS.ipynb](examples/TOPSIS.ipynb): notebook version for learning and experimentation

## Requirements

- Python 3.8+
- `numpy`

Install the dependency with:

```bash
pip install numpy
```

## How to Use

Import the class and provide:

- `decision_matrix`: a 2D list or NumPy array where rows are alternatives and columns are criteria
- `weight_vector`: a 1D list or NumPy array of criterion weights
- `flag`: an optional list showing whether each criterion is a benefit criterion or a cost criterion

### Criterion Types

- `True` means a **benefit criterion**: larger values are better
- `False` means a **cost criterion**: smaller values are better

If you do not provide `flag`, all criteria are treated as benefit criteria by default.

## Quick Example

```python
from models.TOPSIS import TOPSIS

decision_matrix = [
    [7, 9, 9],
    [8, 7, 8],
    [6, 8, 7]
]

weight_vector = [0.3, 0.4, 0.3]
flag = [True, True, False]  # the last criterion is a cost criterion

model = TOPSIS(decision_matrix, weight_vector, flag)

print("Standardized matrix:\n", model.standardized_matrix)
print("Weighted standardized matrix:\n", model.weighted_standardized_matrix)
print("Relative closeness:\n", model.relative_closeness)

# Higher relative closeness means a better alternative.
ranking = model.relative_closeness.argsort()[::-1]
print("Ranking order:", ranking)
```

## Output Explained

After creating a `TOPSIS` object, the class stores several useful results:

- `standardized_matrix`: normalized decision matrix
- `weighted_standardized_matrix`: normalized matrix after applying weights
- `positive_ideal_solution`: squared distance components to the best ideal point
- `negative_ideal_solution`: squared distance components to the worst ideal point
- `relative_closeness`: final TOPSIS score for each alternative

The final score is computed as:

$$
R_i = \frac{d_i^-}{d_i^+ + d_i^-}
$$

where:

- $d_i^+$ is the distance to the positive ideal solution
- $d_i^-$ is the distance to the negative ideal solution

An alternative with a **larger** $R_i$ is preferred.

## Workflow Used in the Code

1. Normalize each criterion column using Euclidean norm.
2. Multiply normalized values by the criterion weights.
3. Compute the positive ideal solution.
4. Compute the negative ideal solution.
5. Calculate the relative closeness score.
6. Rank alternatives from highest score to lowest score.

## Notes for Students

- Make sure the number of criteria in `decision_matrix` matches the number of values in `weight_vector`.
- Use `flag` carefully if your problem contains both benefit and cost criteria.
- This implementation is designed to be easy to read and teach, so it keeps the TOPSIS steps separate.

## Example Interpretation

If two alternatives have scores like:

- Alternative A: `0.82`
- Alternative B: `0.61`

then Alternative A is preferred because it is closer to the ideal solution.

## Acknowledgements

This TOPSIS material was prepared based on knowledge and guidance from Dr. Zhiqiang LIAO.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.

import numpy as np
from models.TOPSIS import TOPSIS

decision_matrix = [
    [9, 7, 6, 7],
    [8, 7, 9, 6],
    [7, 8, 6, 6]
]
weight_vector = [4, 2, 6, 8]

# 1  = benefit criterion (higher is better)
# 0 = cost criterion (lower is better)
flag = [1, 1, 1, 1]

model = TOPSIS(decision_matrix, weight_vector, criteria_type=flag)

R_i = model.relative_closeness
print("\nRelative closeness scores:\n", model.relative_closeness)
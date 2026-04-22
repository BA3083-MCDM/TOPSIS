import numpy as np

class TOPSIS:
    def __init__(self, decision_matrix, weight_vector, flag=None):
        """Initialize data and run all TOPSIS computation steps.

        Parameters:
            decision_matrix: 2D data for alternatives (rows) vs criteria (columns).
            weight_vector: 1D array of criterion weights.
            flag: Optional list where True means benefit criterion (higher is better)
                  and False means cost criterion (lower is better).
        """
        self.decision_matrix = np.array(decision_matrix, dtype=float)
        self.weight_vector = np.array(weight_vector, dtype=float)

        # Number of criteria in decision matrix must match number of weights.
        assert self.decision_matrix.shape[1] == self.weight_vector.shape[0]

        # If not provided, treat all criteria as benefit criteria by default.
        if flag is None:
            flag = [True] * self.decision_matrix.shape[1]
        self.flag = flag

        # Step 1: Normalize each criterion column (vector normalization).
        self.standardized_matrix = self.func_StandardizeDecisionMatrix(self.decision_matrix)
        # Step 2: Multiply normalized values by criterion weights.
        self.weighted_standardized_matrix = self.func_WeightedStandardizedDecisionMatrix(
            self.standardized_matrix,
            self.weight_vector
        )
        # Step 3: Compute squared distances to positive ideal for each criterion.
        self.positive_ideal_solution = self.func_PositiveIdealSolution(
            self.weighted_standardized_matrix,
            self.flag
        )
        # Step 4: Compute squared distances to negative ideal for each criterion.
        self.negative_ideal_solution = self.func_NegativeIdealSolution(
            self.weighted_standardized_matrix,
            self.flag
        )
        # Step 5: Convert distances into relative closeness scores (final ranking basis).
        self.relative_closeness = self.func_IdealSolutionRelativeCloseness(
            positive_matrix = self.positive_ideal_solution,
            negative_matrix = self.negative_ideal_solution
        )

    @classmethod
    def func_StandardizeDecisionMatrix(cls, input_matrix):
        """Normalize each criterion column using Euclidean norm."""
        output_matrix = np.zeros(input_matrix.shape)
        for i in range(input_matrix.shape[1]):
            # Denominator is sqrt(sum(x_ij^2)) for criterion i.
            denom = np.sqrt(np.sum(input_matrix[:, i] ** 2))
            output_matrix[:, i] = input_matrix[:, i] / denom
        return output_matrix

    @classmethod
    def func_WeightedStandardizedDecisionMatrix(cls, standardized_matrix, weight_vector):
        """Apply criterion weights to the normalized matrix."""
        weighted_standardized_matrix = np.zeros(standardized_matrix.shape)
        for i in range(weight_vector.shape[0]):
            weighted_standardized_matrix[:, i] = standardized_matrix[:, i] * weight_vector[i]
        return weighted_standardized_matrix

    @classmethod
    def func_PositiveIdealSolution(cls, input_matrix, flag=None):
        """Build squared-distance components to the positive ideal solution.

        Benefit criterion: ideal is column max.
        Cost criterion: ideal is column min.
        """
        if flag is None:
            flag = [True] * input_matrix.shape[1]

        output_matrix = np.zeros(input_matrix.shape)
        for i in range(input_matrix.shape[1]):
            if flag[i]:
                # Distance to best (max) for benefit criterion.
                output_matrix[:, i] = (input_matrix[:, i] - np.max(input_matrix[:, i])) ** 2
            else:
                # Distance to best (min) for cost criterion.
                output_matrix[:, i] = (input_matrix[:, i] - np.min(input_matrix[:, i])) ** 2
        return output_matrix

    @classmethod
    def func_NegativeIdealSolution(cls, input_matrix, flag=None):
        """Build squared-distance components to the negative ideal solution.

        Benefit criterion: negative ideal is column min.
        Cost criterion: negative ideal is column max.
        """
        if flag is None:
            flag = [True] * input_matrix.shape[1]

        output_matrix = np.zeros(input_matrix.shape)
        for i in range(input_matrix.shape[1]):
            if flag[i]:
                # Distance to worst (min) for benefit criterion.
                output_matrix[:, i] = (input_matrix[:, i] - np.min(input_matrix[:, i])) ** 2
            else:
                # Distance to worst (max) for cost criterion.
                output_matrix[:, i] = (input_matrix[:, i] - np.max(input_matrix[:, i])) ** 2
        return output_matrix

    @classmethod
    def func_IdealSolutionRelativeCloseness(cls, positive_matrix, negative_matrix):
        """Compute TOPSIS score: closeness to ideal and remoteness from nadir.

        R_i = d_i^- / (d_i^+ + d_i^-), where larger R_i is better.
        """
        d_plus = np.zeros(positive_matrix.shape[0])
        d_minus = np.zeros(positive_matrix.shape[0])
        R = np.zeros(positive_matrix.shape[0])
    
        for i in range(positive_matrix.shape[0]):
            # Sum squared components across criteria, then take square root.
            d_plus[i] = np.sqrt(np.sum(positive_matrix[i, :]))
            d_minus[i] = np.sqrt(np.sum(negative_matrix[i, :]))
            R[i] = d_minus[i] / (d_plus[i] + d_minus[i])
        return R

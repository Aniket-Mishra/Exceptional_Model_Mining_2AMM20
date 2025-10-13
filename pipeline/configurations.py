from typing import Dict

class Predicate:
    def __init__(self, feature, op, value):
        self.feature = feature
        self.op = op
        self.value = value

dataset_information = {
    "boston": {
        "file_name": "boston_housing",
        "target": "MEDV",
    }
}

# model_parameters
DEFAULT_RANDOM_STATE = 42  # Answer to all things in the universe
NUM_SPLITS = 5 # for kfold
NUM_ESTIMATORS = 100
NUM_PERMUTATIONS = 1000

# Common parameters
RES_COL = "Residual"  # Shoud be same for all datasets
RES_COL_CV = "Residual_CV"

# conj rune
MAX_RULE_LEN = 3  # Up to us to decide, pikachu, I choose 3
MIN_SUPPORT = 10  # min sg size
N_QUANTILES = (
    6  # numeric thresholds as quantiles 1/Q, 2/Q, ..., (Q-1)/Q
)
BEAM_WIDTH = 200  # Top candi each level to control explosn
TOP_K = 10  # final number of rules to return, K could be 1, but I am not confident enough
EXCLUDE_COLS = {
    RES_COL,
}
CATEGORICAL_AS_OBJECT = True  # object dype as categorical

# poly rune
POLY_BASES: Dict[str, set] = {}
MAX_RULE_LEN = 2

# Dtree
LTREE_MAX_DEPTH = 3
LTREE_MIN_LEAF = 10  # minimum samples in a leaf/subgroup
LTREE_RANDOM_STATE = 42
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

SPLIT_DIRECTIONAL_LISTS = True  # sep list for under/over fit
USE_LOCAL_DISCRETIZATION = True
LOCAL_QUANTILES = 6
LOCAL_THRESHOLDS_PER_FEAT = 4  # keep best k per feature at each branch

# model params
DEFAULT_RANDOM_STATE = 42  # Answer to all things in the universe
NUM_SPLITS = 5  # for kfold
NUM_ESTIMATORS = 100
NUM_PERMUTATIONS = 1000

# Common params
RES_COL = "Residual"  # Shoud be same for all datasets
RES_COL_CV = "Residual_CV"
RES_COL_SIGNED = "Residual_signed"
EXCLUDE_COLS = {
    RES_COL,
    "Residual",
    "Residual_signed",
    "Residual_CV",
    "target",
}

TRADEOFF_METHOD = "pareto"  # we could try res*I(d) or sth like that?

# Threshold quantization for humans
THRESH_RESOLUTION = (
    0.01  # round thresholds to multiples of this before scoring
)

TOP_K_PER_LANGUAGE = 25

# interpretability scoring weights, w4=0 for 0 precision penalty
W_PREDICATES = 1.0
W_OPERATORS = 1.0
W_DEPTH = 1.0
W_PRECISION = 0.0  # not sure how to do this anyway

# for I(d) = exp(-BETA * complexity)
I_DECAY_BETA = 0.7


# conj rune
MAX_RULE_LEN = 5  # Up to us to decide, pikachu, I choose 3
MIN_SUPPORT = 6  # min sg size
# N_QUANTILES = 6  # same as LOCAL_QUANTILES
BEAM_WIDTH = 200  # Top candi each level to control explosn
# TOP_K = 10  # final number of rules to return, K could be 1, but I am not confident enough
# Doing the above for all languages now
# the "Residual" is redundant, but I plan on using other res cols later also
CATEGORICAL_AS_OBJECT = True  # object dype as categorical

# poly rune
POLY_BASES: Dict[str, set] = {}
MAX_RULE_LEN = 2

# tree rune
LTREE_MAX_DEPTH = 4  # 3 gave barely any rules (4-6)
LTREE_MIN_LEAF = 6  # minimum samples in a leaf/subgroup (was 10 b4)
LTREE_RANDOM_STATE = 42
# LTREE_MAX_LEAF_NODES = 30 # hard cap on num leaves, passed as arg we good.

# symb rune
SYMB_MAX_OPS = 3  # prev 3, no change in pareto op
SYMB_MAX_FEATURES = 2  # max 2 distinct features per expr.
SYMB_ALLOW_EQ = False  # permit = in addition to <=, >

# hard cap on total generated expressions after de-dup
SYMB_MAX_EXPRESSIONS = 200
SYMB_TOP_FEATURES = 8  # shortlist features by |corr| with residuals
# quantile thresholds per expression. 25/50/75 when 3
SYMB_THRESHOLDS_PER_EXPR = 3  # same as LOCAL_QUANTILES, use diff val
SYMB_NESTED_TWO_FEATURE = True  # (a+b)/a, (a/b)-a n all
SYMB_ALLOW_TRIPLETS = True  # 3 feature nests. could disable also
SYMB_MAX_TRIPLETS = 40  # so that we don't create like a zillion

# too much
# # cap on # of x operator expressions to generate
# SYMB_MAX_DEEP3 = 20
# SYMB_MAX_DEEP4 = 10

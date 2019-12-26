PAN = False
MAX_PROBLEMS = 0
MIN_PROBLEMS = 0
N_CORE = 16
MULTICORE = True
MULTICLASSIFIER = True
S = STORAGE = True

if PAN:
    MAX_PROBLEMS = 0
    MIN_PROBLEMS = 0
    N_CORE = 16
    S = STORAGE = False
    MULTICORE = False

weights = {
    'english':  None,  # [1.25, 1, 1, 1, .75, .75],
    'french':   None,
    'italian':  None,
    'spanish':  None
}

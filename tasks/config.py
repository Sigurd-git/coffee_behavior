# Common configuration for all tasks

# Screen settings
SCREEN_DISTANCE = 57  # Distance from screen (cm)
SCREEN_WIDTH = 30     # Physical screen width (cm)

# Timing parameters
INSTRUCTION_DURATION = 2.0
FEEDBACK_DURATION = 1.0
MIN_ISI = 1.0
MAX_ISI = 2.0

# Data saving
DATA_DIR = "data"
RESULTS_FORMAT = "csv"  # or "mat" for MATLAB compatibility

# Response keys
RESPONSE_KEYS = {
    'confirm': 'space',
    'quit': 'escape',
    'number_keys': ['1', '2', '3', '4', '5', '6', '7', '8', '9']
} 
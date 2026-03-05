# config.py (NEW FILE — create at the root of mean_reversion/)
# FIX 5.2: Single source of truth for all parameters

BIAS_THRESHOLD = 1.5
VOL_PERIOD = 20
SKEW_PERIOD = 5
WARMUP_DAYS = 45
HURST_WINDOW = 200
SMA_PERIOD = 20
HURST_TAUS = [4, 8, 16, 32, 64]
RISK_PCT = 0.01
MNQ_POINT_VALUE = 2.0
# Data Paths
DATA_FILES = {'NQ': 'ENQH26.csv', 'ES': 'EPH26.csv', 'YM': 'YMH26.csv', 'RTY': 'RTYH26.csv', 'CL': 'CLEG26.csv', 'GC': 'GCEG26.csv'}

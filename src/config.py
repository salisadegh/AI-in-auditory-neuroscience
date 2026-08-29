"""Global configuration: paths, signal-processing constants, and the analysis seed."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ABR_DATA", ROOT / "data"))
EARNDB_DIR = DATA / "earndb"
MENDELEY_DIR = DATA / "mendeley"
LABELS = ROOT / "labels"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED = 42

# earndb: 48 kHz, epoch 0-41.7 ms after stimulus onset
EARNDB_FS = 48000
EARNDB_RESPONSE_MS = (4.0, 16.0)
EARNDB_NOISE_MS = (25.0, 41.0)
EARNDB_ANALYSIS_MS = (0.0, 20.0)

# Mendeley (IHS SmartEP): 20 kHz, shorter-latency click responses
MENDELEY_FS = 20000
MENDELEY_RESPONSE_MS = (2.0, 10.0)
MENDELEY_NOISE_MS = (12.0, 22.0)

# ABR analysis band for engineered features
BAND_HZ = (100.0, 1500.0)
BAND_ORDER = 4

# Two-replication reproducibility criterion
REPRODUCIBILITY_CUT = 0.25

# Waveform length supplied to waveform-input detectors
RAW_LENGTH = 220

BOOTSTRAP_CI = 2000
BOOTSTRAP_TEST = 4000

from pathlib import Path

ROOT = Path(__file__).parent
PARAMS = ROOT / "config.yaml"

DATA = ROOT / "assets" / "samples"
DATA.mkdir(parents=True, exist_ok=True)

TRAINED_MODELS = ROOT / "trained"
TRAINED_MODELS.mkdir(parents=True, exist_ok=True)

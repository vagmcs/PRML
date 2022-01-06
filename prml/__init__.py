# Standard Library
import os
from pathlib import Path

root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = root_dir.parent / "datasets"

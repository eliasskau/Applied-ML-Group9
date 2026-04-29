import os
from pathlib import Path
import numpy as np
import pandas as pd

total_count = 0
base_path = Path("../data/images")
for emotion_dir in base_path.iterdir():
    if emotion_dir.is_dir():
        count = len(list(emotion_dir.rglob("*.jpg")))
        print(f"{emotion_dir.name}: {count} images")
        total_count = total_count + count

print(f"total: {total_count} images ")


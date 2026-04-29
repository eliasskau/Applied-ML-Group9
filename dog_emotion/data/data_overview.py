import os
from pathlib import Path
import numpy as np
import pandas as pd

"""
data_overview.py
this .py file is a script that when ran will produce basic preliminary data 
statistics on the data in the projects DVC. These statistics are to be used 
in the applied machine learning proposal.

These statistics include (to be expanded)
 - Number of images in each category + total number of images
 

This will have to be updated as data is added unless new data conforms to 
the same folder structure and just adds .jpg files.

"""

total_count = 0
base_path = Path("../../data/images")
for emotion_dir in base_path.iterdir():
    if emotion_dir.is_dir():
        count = len(list(emotion_dir.rglob("*.jpg")))
        print(f"{emotion_dir.name}: {count} images")
        total_count = total_count + count

print(f"total: {total_count} images ")


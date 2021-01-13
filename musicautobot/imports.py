from pathlib import Path
import music21
import numpy as np

from fastai.vision import get_files, download_url
from fastai.basics import load_data, untar_data
from fastai.text.models import TransformerXL
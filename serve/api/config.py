"""
Global Flask Application Setting

See `.flaskenv` for default settings.
 """

import os
# from app import app
from . import app
from pathlib import Path

class Config(object):
    project_path = Path(__file__).parents[2]
    LIB_PATH = project_path
    DATA_PATH = project_path/'data/numpy'
    DATA_SAVE_NAME = 'musicitem_data_save.pkl'
    MULTITASK_MODEL_PATH = DATA_PATH/'pretrained/MultitaskSmallKeyC.pth'
    MUSIC_MODEL_PATH = DATA_PATH/'pretrained/MusicTransformerKeyC.pth'

app.config.from_object('api.config.Config')
app.config.from_pyfile('api.cfg')

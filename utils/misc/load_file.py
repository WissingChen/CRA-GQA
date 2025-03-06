"""
load file & save file
"""
import numpy as np
import torch
import pandas as pd
import json
import yaml


def load_file(path, kwargs=None):
    _type = path.split(".")[-1]
    if _type == "json":
        file = _load_json(path, kwargs)
    elif _type == "yml":
        file = _load_yml(path, kwargs)
    return file

def _load_csv(path, kwargs):
    pass

def _load_json(path, kwargs):
    pass

def _load_yml(path, kwargs):
    file = yaml.load(open(path), Loader=yaml.FullLoader)
    return file

# Generated by chatGPT - automatically import all modules in this folder

import os
import importlib

_model_list = dict()
def register_model(name, cls):
    _model_list[name] = cls

def list_models(lower=False):
    return { k.lower(): v for k,v in _model_list.items() } if lower else _model_list

def get_model(name):
    return list_models(lower=True).get(name.lower(), None)

def load_model(model_path):
    from models.base import ModelBase
    model = ModelBase().load(model_path)
    return model

# Dynamically generate the class mapping
package_path = os.path.dirname(__file__)
for file_name in os.listdir(package_path):
    if file_name.endswith(".py") and file_name != "__init__.py":
        module_name = file_name[:-3]
        module = importlib.import_module(f"{__name__}.{module_name}")
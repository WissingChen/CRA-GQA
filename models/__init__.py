# build a model list
from .cra import CRA
from .tempclip import TempCLIP
model_fns = {"tempclip": TempCLIP, "CRA": CRA}

# build a pipeline
from .tempclip_pipeline import TempCLIPPipeline
from .cra_pipeline import CRAPipeline
pipeline_fns = {"tempclip": TempCLIPPipeline,
                "cra":CRAPipeline}
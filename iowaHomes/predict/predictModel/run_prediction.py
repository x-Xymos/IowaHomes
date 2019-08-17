try:
    from .main import run_prediction as run_prediction_
    from .main import models
except:
    from main import run_prediction as run_prediction_
    from main import models

def run_prediction(args):

    return run_prediction_(models, args)
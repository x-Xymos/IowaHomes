from iowaHomes.predict.predictModel.run_prediction import run_prediction
from iowaHomes.predict.templates.predict.feature_element_def import elements
from iowaHomes.predict.predictModel.main import engineered_features
from iowaHomes.predict.predictModel.score_models import score_models

import pickle, os, random

def test_run_prediction(n=10):
    """
    Runs the run_predicition function n number of times, using a random value
    for the user_input every time the function runs

    Expected result: Function runs N times with no errors

    :return None:
    """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FOPATH = os.path.join(BASE_DIR, 'iowaHomes/predict/predictModel/predictionModels/training_data/featureOrder.sav')

    featureOrder = pickle.load(open(FOPATH, 'rb'))

    target_features = []
    for feat in featureOrder:
        target_features.append(feat)
        for e_feat in engineered_features:
            if e_feat['name'] == feat:
                target_features.remove(feat)
                for dep in e_feat['dependencies']:
                    target_features.append(dep)

    for x in range(n):
        args = {}
        for feat in target_features:
            if elements[feat]['type'] == "dropdown":
                argsFeatFieldsLen = elements[feat]['fields'].__len__()
                args[feat] = elements[feat]['fields'][random.randint(0,argsFeatFieldsLen-1)]['value']
            elif elements[feat]['type'] == "slider":
                min = int(elements[feat]['min'])
                max = int(elements[feat]['max'])
                args[feat] = random.randint(min, max)

        print(run_prediction(args))

def test_score_models():
    """
       Runs the score_models function that outputs the result of different
       metrics that measure the accuracy of all the models

       Expected result: Function runs and prints the results of all the models

       :return None:
       """
    score_models()

def run_tests():
    test_score_models()
    test_run_prediction(15)

run_tests()
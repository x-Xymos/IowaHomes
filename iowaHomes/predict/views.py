from django.views import generic
from django.shortcuts import render
import datetime, os, sys, pickle

PROJECT_DIR = os.path.dirname(__file__)
MPATH = os.path.join(PROJECT_DIR, "predictModel/")
sys.path.append(MPATH)
TPATH = os.path.join(PROJECT_DIR, "templates/predict")
sys.path.append(TPATH)

from run_prediction import run_prediction
from main import engineered_features


#most of these view functions and their html templates should be rewritten to use a template
#that replaces the codeblock inside the template with the content
class IndexView(generic.ListView):
    template_name = 'predict/index.html'

    def get_queryset(self):
        return 'predict/index.html'

class BrowseView(generic.ListView):
    template_name = 'predict/browseHomes.html'

    def get_queryset(self):
        return 'predict/browseHomes.html'

class LoginView(generic.ListView):
    template_name = 'predict/login.html'

    def get_queryset(self):
        return 'predict/login.html'


def EstimateView(request):
    """
    Renders the estimate page

    Parameters
    ----------
    :param request : dictionary

    Returns
    ----------
    render : HttpResponse

    """

    from feature_element_def import elements


    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FOPATH = os.path.join(BASE_DIR, 'predict/predictModel/predictionModels/training_data/featureOrder.sav')
    featureOrder = pickle.load(open(FOPATH, 'rb'))

    target_features = []
    # converting features that were engineered, into their dependencies
    for feat in featureOrder:
        target_features.append(feat)
        for e_feat in engineered_features:
            if e_feat['name'] == feat:
                target_features.remove(feat)
                for dep in e_feat['dependencies']:
                    target_features.append(dep)

    target_elements = []
    for feat in target_features:
        target_elements.append(elements[feat])


    context = {'pred_args': {},
               'target_elements': target_elements,
               'prediction': None,

               }

    for key, value in request.POST.items():
        context['pred_args'][key] = value

    print(context['pred_args'])
    if len(context['pred_args']) > 0:
        del context['pred_args']['csrfmiddlewaretoken']
        context['prediction'] = int(run_prediction(context['pred_args']))


    return render(request, 'predict/estimate.html', context)


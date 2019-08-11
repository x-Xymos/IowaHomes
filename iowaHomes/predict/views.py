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

class IndexView(generic.ListView):
    template_name = 'predict/index.html'
    #return render(request, 'predict/index.html')

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


def range_slider_factory(slider):

    r = '<div class="rowTab">' \
        '   <script> ' \
        '       $( function() {' \
        '$("#hidden{{slider.name}}").val({{slider.value}});' \
        '       $( "#slider{{slider.name}}" ).slider({' \
        '       value:{{slider.value}},' \
        '       min: {{slider.min}},' \
        '       max: {{slider.max}},' \
        '       step: {{slider.step}},' \
        '       slide: function( event, ui ) {' \
        '           $("#amount{{slider.name}}" ).val(ui.value + " {{slider.unit}}");' \
        '           $("#hidden{{slider.name}}").val(ui.value);' \
        '           }' \
        '         });' \
        '         $( "#amount{{slider.name}}" ).val( $("#slider{{slider.name}}" ).slider( "value" ) );' \
        '} );' \
        '</script>' \
        '<div class="input-label">' \
        '   <label class="rangeLabel" for="amount{{slider.name}}">{{slider.labelText}}</label>' \
        '   </div>' \
        '   <div class="input-field" title="{{slider.tooltip}}">' \
                '<div class="tooltip-box" title="{{slider.tooltip}}"></div>'\
        '       <div class="range-slider" id="slider{{slider.name}}"></div>' \
        '       <input class="range-amount-text" type="text" id="amount{{slider.name}}" readonly style="border:0; font-weight:bold;">' \
        '       <input type="hidden" name="{{slider.name}}" id="hidden{{slider.name}}" value="">' \
        '   </div>' \
        '</div>'
    return r.replace("{{slider.name}}",slider['name'])\
        .replace("{{slider.value}}",slider['value'])\
        .replace("{{slider.min}}",slider['min'])\
        .replace("{{slider.max}}",slider['max'])\
        .replace("{{slider.step}}",slider['step']) \
        .replace("{{slider.unit}}", slider['unit']) \
        .replace("{{slider.labelText}}",slider['labelText'])\
        .replace("{{slider.tooltip}}",slider['tooltip'])\


def dropdown_menu_factory(menu):
    # r = '<div class="rowTab">' \
    #     '   <script>' \
    #     '       $( function() {' \
    #     '           $( document ).tooltip({ });' \
    #     '       } );' \
    #     '   </script>' \
    #     '   <div class="input-label">' \
    #     '       <label id="{{menu.name}}-label" for="{{menu.name}}">{{menu.labelText}}</label>' \
    #     '   </div>' \
    #     '   <div class="input-field" title="{{menu.tooltip}}">' \
    #             '<div class="tooltip-box" title="{{slider.tooltip}}"></div>' \
    #     '       <select class="input-dropdown" id={{menu.name}} name="{{menu.name}}">'

    r = '<div class="rowTab">' \
        '   <div class="input-label">' \
        '       <label id="{{menu.name}}-label" for="{{menu.name}}">{{menu.labelText}}</label>' \
        '   </div>' \
        '   <div class="input-field" title="{{menu.tooltip}}">' \
        '<div class="tooltip-box" title="{{slider.tooltip}}"></div>' \
        '       <select class="input-dropdown" id={{menu.name}} name="{{menu.name}}">'

    for field in menu['fields']:
        r = r + '<option value="%s">%s</option>' %(field['value'], field['text'])


    r = r + "</select>" \
            "</div>" \
            "</div>"
    return r.replace('{{menu.name}}', menu['name'])\
            .replace('{{menu.labelText}}', menu['labelText'])\
            .replace('{{menu.tooltip}}', menu['tooltip'])\


def EstimateView(request):
    from feature_element_def import elements


    elem_factory_def = {'dropdown': dropdown_menu_factory,
                        'slider': range_slider_factory}


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

    cooked_elems = []
    for feat in target_features:
        c_elem = elem_factory_def[elements[feat]['type']](elements[feat])
        cooked_elems.append(c_elem)
        # for elem in elements:
        #     if feat == elem['name']:
        #         c_elem = elem_factory_def[elem['type']](elem)
        #         cooked_elems.append(c_elem)
        #         continue


    context = {'pred_args': {},
               'cooked_elems': cooked_elems,
               'prediction': None,

               }

    for key, value in request.POST.items():
        context['pred_args'][key] = value

    print(context['pred_args'])
    if len(context['pred_args']) > 0:
        del context['pred_args']['csrfmiddlewaretoken']
        context['prediction'] = int(run_prediction(context['pred_args']))


    return render(request, 'predict/estimate.html', context)


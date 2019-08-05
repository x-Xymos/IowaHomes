from django.views import generic
from django.shortcuts import render
import datetime
import pickle
import pandas as pd
import os

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


# def image(request):
#     INK = "red", "blue", "green", "yellow"
#     # ... create/load image here ...
#     image = Image.new("RGB", (128, 128), random.choice(INK))
#
#     buff = BytesIO()
#     image.save(buff, format="PNG")
#     img_str = base64.b64encode(buff.getvalue())
#
#     data_uri = b'data:image/jpg;base64,'
#     data_uri = data_uri + img_str
#
#     return data_uri.decode('utf-8')

def range_slider_factory(slider):

    r = '<div class="rowTab">' \
        '   <script> ' \
        '       $( function() {' \
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
    r = '<div class="rowTab">' \
        '   <script>' \
        '       $( function() {' \
        '           $( document ).tooltip();' \
        '       } );' \
        '   </script>' \
        '   <div class="input-label">' \
        '       <label id="{{menu.name}}-label" for="{{menu.name}}">{{menu.labelText}}</label>' \
        '   </div>' \
        '   <div class="input-field" title="{{menu.tooltip}}">' \
        '       <select class="input-dropdown" id={{menu.name}} name="{{menu.name}}">'
    for field in menu['fields']:
        r = r + '<option value="%s">%s</option>' %(field['value'], field['text'])

    r = r + "</select>" \
            "</div>" \
            "</div>"
    return r.replace('{{menu.name}}', menu['name'])\
            .replace('{{menu.labelText}}', menu['labelText'])\
            .replace('{{menu.tooltip}}', menu['tooltip'])\



def runEstimate(args):
    models = ['l_reg_model.sav', 'lasso_model.sav','ridge_model.sav','b_ridge_model.sav']
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    df = pd.DataFrame(columns=list(args.keys()))
    df.loc[0] = list(args.values())
    
    pred = 0
    for model in models:
        path = os.path.join(BASE_DIR, 'iowaHomes/predictionModels/28features/')

        reg = pickle.load(open(path + model, 'rb'))
        pred = pred + reg.predict(df)


    return pred / len(models)


def EstimateView(request):

    dropdown_menus = [
        {"name":"MSZoning",
            "fields": [{"value":"A","text":"Agriculture"},
                        {"value": "C", "text": "Commercial"},
                        {"value": "FV", "text": "Floating Village"},
                        {"value": "I", "text": "Industrial"},
                        {"value": "RH", "text": "Residential - High Density"},
                        {"value": "RM", "text": "Residential - Medium Density"},
                        {"value": "RP", "text": "Residential - Low Density Park"},
                        {"value": "RL", "text": "Residential - Low Density"},
                        ],
            "labelText":"Zoning Classification",
            "tooltip":"Identifies the general zoning classification"},

        {"name": "OverallQual",
             "fields": [{"value": "10", "text": "Very Excellent"},
                        {"value": "9", "text": "Excellent"},
                        {"value": "8", "text": "Very Good"},
                        {"value": "7", "text": "Good"},
                        {"value": "6", "text": "Above Average"},
                        {"value": "5", "text": "Average"},
                        {"value": "4", "text": "Below Average"},
                        {"value": "3", "text": "Fair"},
                        {"value": "2", "text": "Poor"},
                        {"value": "1", "text": "Very Poor"},
                        ],
             "labelText": "Overall Quality",
             "tooltip": "Rates the overall material and finish of the house"},

        {"name": "OverallCond",
             "fields": [{"value": "10", "text": "Very Excellent"},
                        {"value": "9", "text": "Excellent"},
                        {"value": "8", "text": "Very Good"},
                        {"value": "7", "text": "Good"},
                        {"value": "6", "text": "Above Average"},
                        {"value": "5", "text": "Average"},
                        {"value": "4", "text": "Below Average"},
                        {"value": "3", "text": "Fair"},
                        {"value": "2", "text": "Poor"},
                        {"value": "1", "text": "Very Poor"},
                        ],
             "labelText": "Overall Condition",
             "tooltip": " Rates the overall condition of the house"},

        {"name": "MasVnrType",
             "fields": [{"value": "BrkCmn", "text": "Brick Common"},
                        {"value": "BrkFace", "text": "Brick Face"},
                        {"value": "CBlock", "text": "Cinder Block"},
                        {"value": "Stone", "text": "Stone"},
                        {"value": "None", "text": "None"},
                        ],
             "labelText": "Masonry Type",
             "tooltip": "Masonry veneer type"},

        {"name": "ExterCond",
             "fields": [{"value": "Ex", "text": "Excellent"},
                        {"value": "Gd", "text": "Good"},
                        {"value": "TA", "text": "Average/Typical"},
                        {"value": "Fa", "text": "Fair"},
                        {"value": "Po", "text": "Poor"},
                        ],
             "labelText": "Exterior Condition",
             "tooltip": "Evaluates the present condition of the material on the exterior"},

        {"name": "BsmtQual",
         "fields": [{"value": "Ex", "text": "Excellent (100+ inches)"},
                    {"value": "Gd", "text": "Good (90-99 inches)"},
                    {"value": "TA", "text": "Typical (80-89 inches)"},
                    {"value": "Fa", "text": "Fair (70-79 inches)"},
                    {"value": "Po", "text": "Poor (<70 inches)"},
                    {"value": "NA", "text": "No Basement"},
                    ],
         "labelText": "Basement Height",
         "tooltip": "Evaluates the height of the basement"},

        {"name": "BsmtExposure",
         "fields": [{"value": "Gd", "text": "Good Exposure"},
                    {"value": "Av", "text": "Average Exposure (split levels or foyers typically score average or above)"},
                    {"value": "Mn", "text": "Mimimum Exposure"},
                    {"value": "No", "text": "No Exposure"},
                    {"value": "NA", "text": "No Basement"},
                    ],
         "labelText": "Basement Exposure",
         "tooltip": "Refers to walkout or garden level walls"},

        {"name": "BsmtFinType1",
         "fields": [{"value": "GLQ", "text": "Good Living Quarters"},
                    {"value": "ALQ", "text": "Average Living Quarters"},
                    {"value": "BLQ", "text": "Below Average Living Quarters"},
                    {"value": "Rec", "text": "Average Rec Room"},
                    {"value": "LwQ", "text": "Low Quality"},
                    {"value": "Unf", "text": "Unfinshed"},
                    {"value": "NA", "text": "No Basement"},
                    ],
         "labelText": "Basement Finished Area Rating",
         "tooltip": " Rating of basement finished area"},

        {"name": "HeatingQC",
         "fields": [{"value": "Ex", "text": "Excellent"},
                    {"value": "Gd", "text": "Good"},
                    {"value": "TA", "text": "Average/Typical"},
                    {"value": "Fa", "text": "Fair"},
                    {"value": "Po", "text": "Poor"},
                    ],
         "labelText": "Heating quality",
         "tooltip": " Heating quality and condition"},

        {"name": "CentralAir",
         "fields": [{"value": "Y", "text": "Yes"},
                    {"value": "N", "text": "No"},
                    ],
         "labelText": "Central air",
         "tooltip": "Central air conditioning"},

        {"name": "KitchenQual",
         "fields": [{"value": "Ex", "text": "Excellent"},
                    {"value": "Gd", "text": "Good"},
                    {"value": "TA", "text": "Average/Typical"},
                    {"value": "Fa", "text": "Fair"},
                    {"value": "Po", "text": "Poor"},
                    ],
         "labelText": "Kitchen quality",
         "tooltip": "Kitchen quality rating"},

        {"name": "Functional",
         "fields": [{"value": "Typ", "text": "Typical Functionality"},
                    {"value": "Min1", "text": "Minor Deductions 1"},
                    {"value": "Min2", "text": "Minor Deductions 2"},
                    {"value": "Mod", "text": "Moderate Deductions"},
                    {"value": "Maj1", "text": "Major Deductions 1"},
                    {"value": "Maj2", "text": "Major Deductions 2"},
                    {"value": "Sev", "text": "Severely Damaged"},
                    {"value": "Sal", "text": "Salvage only"},
                    ],
         "labelText": "Functionality Rating",
         "tooltip": "Home functionality (Assume typical unless deductions are warranted)"},

        {"name": "PavedDrive",
         "fields": [{"value": "Y", "text": "Paved"},
                    {"value": "P", "text": "Partial Pavement"},
                    {"value": "N", "text": "Dirt/Gravel"},
                    ],
         "labelText": "Driveway type",
         "tooltip": ""},

    ]

    range_sliders = [
        {"name":"LotArea",
         "min": '1000',
         "max": '50000',
         "step": '50',
         "value": '1000',
         "unit": "sq. ft",
         "labelText": "Lot Area",
         "tooltip": "Lot size in square feet"
         },

        {"name": "YearBuilt",
         "min": '1870',
         "max": str(datetime.datetime.today().year),
         "step": '1',
         "value": str(datetime.datetime.today().year),
         "unit": "",
         "labelText": "Year Built",
         "tooltip": "Original construction date"
         },

        {"name": "MasVnrArea",
         "min": '0',
         "max": '5000',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "Masonry Area",
         "tooltip": " Masonry veneer area in square feet"
         },

        {"name": "TotalBsmtSF",
         "min": '0',
         "max": '8000',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "Basement Area",
         "tooltip": "Total square feet of basement area"
         },

        {"name": "YearRemodAdd",
         "min": '1870',
         "max": str(datetime.datetime.today().year),
         "step": '1',
         "value": str(datetime.datetime.today().year),
         "unit": "",
         "labelText": "Remodel/Addition Year",
         "tooltip": "Remodel date (same as construction date if no remodeling or additions)"
         },

        {"name": "GrLivArea",
         "min": '200',
         "max": '7000',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "Above grade area",
         "tooltip": "Above grade (ground) living area square feet"
         },

        {"name": "BsmtFullBath",
         "min": '0',
         "max": '10',
         "step": '1',
         "value": '0',
         "unit": "",
         "labelText": "Basement full bathrooms",
         "tooltip": "Amount of full basement bathrooms"
         },

        {"name": "BedroomAbvGr",
         "min": '0',
         "max": '15',
         "step": '1',
         "value": '0',
         "unit": "",
         "labelText": "Bedrooms above grade",
         "tooltip": "Bedrooms above grade (does NOT include basement bedrooms)"
         },

        {"name": "KitchenAbvGr",
         "min": '0',
         "max": '15',
         "step": '1',
         "value": '0',
         "unit": "",
         "labelText": "Kitchens",
         "tooltip": "Kitchens above grade"
         },

        {"name": "Fireplaces",
         "min": '0',
         "max": '15',
         "step": '1',
         "value": '0',
         "unit": "",
         "labelText": "Fireplaces",
         "tooltip": "Number of fireplaces"
         },

        {"name": "GarageYrBlt",
         "min": '1870',
         "max": str(datetime.datetime.today().year),
         "step": '1',
         "value": str(datetime.datetime.today().year),
         "unit": "",
         "labelText": "Garage Year Built",
         "tooltip": "Year garage was built"
         },

        {"name": "GarageCars",
         "min": '0',
         "max": '15',
         "step": '1',
         "value": '0',
         "unit": "",
         "labelText": "Garage Capacity",
         "tooltip": "Size of garage in car capacity"
         },

        {"name": "GarageArea",
         "min": '0',
         "max": '3000',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "Garage Area",
         "tooltip": "Size of garage in square feet"
         },

        {"name": "WoodDeckSF",
         "min": '0',
         "max": '2000',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "Wood deck area",
         "tooltip": "Wood deck area in square feet"
         },

        {"name": "OpenPorchSF",
         "min": '0',
         "max": '1250',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "Open Porch Area",
         "tooltip": "Open porch area in square feet"
         },

        {"name": "EnclosedPorch",
         "min": '0',
         "max": '1000',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "Enclosed porch area",
         "tooltip": "Enclosed porch area in square feet"
         },

        {"name": "3SsnPorch",
         "min": '0',
         "max": '1000',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "Three season porch area",
         "tooltip": "Three season porch area in square feet"
         },

        {"name": "ScreenPorch",
         "min": '0',
         "max": '1000',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "Screen porch area",
         "tooltip": "Screen porch area in square feet"
         },

        {"name": "1stFlrSF",
         "min": '200',
         "max": '7000',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "First Floor Area",
         "tooltip": "First Floor area in square feet"
         },

        {"name": "2ndFlrSF",
         "min": '0',
         "max": '5000',
         "step": '5',
         "value": '0',
         "unit": "sq. ft",
         "labelText": "Second Floor Area",
         "tooltip": "Second Floor area in square feet"
         },

    ]

    elem_render_order = ['MSZoning',
                     'OverallQual',
                     'OverallCond',
                     'ExterCond',
                     'YearBuilt',
                     'YearRemodAdd',
                     'LotArea',
                     'GrLivArea',
                     '1stFlrSF',
                     '2ndFlrSF',
                     'BsmtQual',
                     'BsmtExposure',
                     'BsmtFinType1',
                     'BsmtFullBath',
                     'TotalBsmtSF',
                     'HeatingQC',
                     'CentralAir',
                     'BedroomAbvGr',
                     'KitchenAbvGr',
                     'KitchenQual',
                     'Functional',
                     'Fireplaces',
                     'GarageYrBlt',
                     'GarageCars',
                     'GarageArea',
                     'PavedDrive',
                     'WoodDeckSF',
                     'OpenPorchSF',
                     'EnclosedPorch',
                     '3SsnPorch',
                     'ScreenPorch',
                     'MasVnrType',
                     'MasVnrArea',

                     ]

    cooked_elems = []
    for elem in elem_render_order:
        for menu in dropdown_menus:
            if elem == menu['name']:
                cooked_elems.append(dropdown_menu_factory(menu))
                continue

        for slider in range_sliders:
            if elem == slider['name']:
                cooked_elems.append(range_slider_factory(slider))
                continue


    context = {'pred_args': {},
               'cooked_elems': cooked_elems,
               'prediction': None,

               }

    for key, value in request.POST.items():
        context['pred_args'][key] = value

    if len(context['pred_args']) > 0:
        del context['pred_args']['csrfmiddlewaretoken']
        context['prediction'] = runEstimate(context['pred_args'])

    print(context['pred_args'])
    return render(request, 'predict/estimate.html', context)

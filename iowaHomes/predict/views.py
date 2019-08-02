from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    return render(request, 'predict/index.html')


def run_prediction(request):
    context = {'pred_result': {}}
    for key, value in request.POST.items():
        context['pred_result'][key] = value
    print(context)
    #context = {'pred_result': {"text":request.POST.get('choice','None')}}
    return render(request, 'predict/index.html', context)

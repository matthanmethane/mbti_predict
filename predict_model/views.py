from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm
from .chat_conversion.chat_json_to_csv import chat_json_to_csv
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from .ml_model.logistic_regressor import predict_mbti


def index(request):
    return HttpResponse("Hello, world. You're at the MBTI Predict index.")

def upload_file(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        df_a, df_b = chat_json_to_csv(uploaded_file)
        print(df_a.head())
        print(predict_mbti(df_a))
        print(predict_mbti(df_b))
        # fs = FileSystemStorage()
        # name = fs.save(uploaded_file.name, uploaded_file)
        # context['url'] = fs.url(name)
        return HttpResponse("Hello, world. You're at the MBTI Predict index.")

    return render(request, 'upload.html', context)
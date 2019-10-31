from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import math
import cv2 as cv

from rest_framework import viewsets
from .models import Histories, Fishes
from .serializers import HistoriesSerializer, FishesSerializer

# Create your views here.

def home(request):
    return render(request,'home.html')

def admindash(request):
    return render(request,'admin.html')

def datatables(request):
    return render(request,'datatables.html')

def elements(request):
    return render(request,'elements.html')

def datasets(request):
    return render(request,'datasets.html')

def add(request):

    val1=int(request.POST['num1'])
    val2=int(request.POST['num2'])
    res= val1+val2

    return render(request,'result.html',{'result':res})

def showcase(request):
    return render(request,'showcase.html')

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['photo']
        fs = FileSystemStorage()
        fs.save(uploaded_file.name,uploaded_file)
    
    return render(request,'showcase.html',{'name':uploaded_file.name, 'size':uploaded_file.size})

# def segmentation(request):
    

def tes(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['photo']
        fs = FileSystemStorage()
        
        name = fs.save(uploaded_file.name,uploaded_file)
        url = fs.url(name)

    trained_model = load_model('D:/FIXTA/FishRecognitionVGG19data50.h5')

    classes = ['acanthaluteres_vittiger',
    'acanthistius_cinctus',
    'acanthopagrus_berda',
    'aluterus_monoceros',
    'amphiprioninae',
    'anampses_caeruleopunctatus',
    'anampses_melanurus',
    'anampses_twistii',
    'anodontostoma_chacunda',
    'bodianus_axillaris',
    'bodianus_bilunulatus',
    'choerodon_fasciatus',
    'choerodon_graphicus',
    'choerodon_venustus',
    'chromileptes_altivelis',
    'coris_gaimard',
    'coris_picta',
    'epinephelus_howlandi',
    'epinephelus_maculatus',
    'gnathodentex_aureolineatus',
    'gracila_albomarginata',
    'gymnosarda_unicolor',
    'labroides_dimidiatus',
    'liopropoma_mitratum',
    'liopropoma_susumi',
    'lutjanus_kasmira',
    'lutjanus_sebae',
    'macropharyngodon_kuiteri',
    'mugim_cephalus',
    'nemipterus_hexodon',
    'ophthalmolepis_lineolatus',
    'oxymonacanthus_longirostris',
    'paraluteres_prionurus',
    'pervagor_melanocephalus',
    'plectranthias_nanus',
    'pseudanthias_bicolor',
    'pseudanthias_pleurotaenia',
    'pseudocheilinus_hexataenia',
    'pseudojuloides_cerasinus',
    'pteragogus_cryptus',
    'rastrelliger_kanagurta',
    'sarda_orientalis',
    'scaevius_milii',
    'scolopsis_vosmeri',
    'serranocirrhitus_latus',
    'symphorichthys_spilurus',
    'thalassoma_hardwicke',
    'thalassoma_nigrofasciatum',
    'triaenodon_obesus',
    'wetmorella_nigropinnata']

    img_path="D:/fishrecognition/fishrecognition/"+ str(url)
    k=72
    img = image.load_img(img_path, target_size=(100, 250))
    img = img.resize((250, 100))
    test_image = image.img_to_array(img)
    #     plot.imshow(test_image/256)
    test_image = np.expand_dims(test_image, axis=0)
    result = trained_model.predict(test_image)
    hasil = ''
    species = ''
    percentage = ''

    for i in range(len(classes)):
        if result[0][i] * 100 > k:
            if i == 0:
                hasil = str(classes[i]) + '\t\t:' + str(math.floor(result[0][i] * 100)) + '%'
                species = str(classes[i])
                percentage = str(math.floor(result[0][i] * 100))
            elif i > 0:
                if result[0][i] * 100 > result[0][i - 1] * 100:
                    hasil = str(classes[i]) + '\t\t:' + str(math.floor(result[0][i] * 100)) + '%'
                    species = str(classes[i])
                    percentage = int(math.floor(result[0][i] * 100))

    if hasil == '':
        return render(request,'showcase.html',{'result':'unknown','img':url,'species':'unknown','percentage':0})
    else:
        return render(request,'showcase.html',{'result':hasil,'img':url,'species':species,'percentage':percentage})

class HistoriesView(viewsets.ModelViewSet):
    queryset = Histories.objects.all()
    serializer_class = HistoriesSerializer

class FishesView(viewsets.ModelViewSet):
    queryset = Fishes.objects.all()
    serializer_class = FishesSerializer
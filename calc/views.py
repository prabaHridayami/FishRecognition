from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from keras.models import *
from keras.applications import *
import numpy as np
from keras.preprocessing import image
import math
import cv2
from PIL import Image 
from matplotlib import pyplot as plt 

from rest_framework import viewsets
from .models import Histories, Fishes, Datasets
from .serializers import HistoriesSerializer, FishesSerializer

# Create your views here.


def home(request):
    return render(request,'home.html')

def admindash(request):
    return render(request,'admin.html')

def training(request):
    return render(request,'training.html')

def datasets(request):
    return render(request,'datasets.html')
    
def setting(request):
    return render(request,'setting.html')

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

def objectDetection(url, name):
    img = cv2.imread(url)
    blurred = cv2.blur(img, (3,3))
    canny = cv2.Canny(blurred, 50, 300)

    ## find the non-zero min-max coords of canny
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)

    ## crop the region
    cropped = img[y1-10:y2+10, x1-10:x2+10]
    # tagged = cv2.rectangle(img.copy(), (x1-10,y1-10), (x2+10,y2+10), (0,255,0), 3, cv2.LINE_AA)
    # cv2.imshow("canny", canny)
    # cv2.imshow("tagged", tagged)
    url = "D:/fishrecognition/fishrecognition/media/cropped/"+name
    cv2.imwrite(url, cropped)

    return url

def bgRemover(url,name):

    image = cv2.imread(url) 
    mask = np.zeros(image.shape[:2], np.uint8) 
    
    backgroundModel = np.zeros((1, 65), np.float64) 
    foregroundModel = np.zeros((1, 65), np.float64) 
    
    rectangle = (20, 20, 550, 300) 
    
    cv2.grabCut(image, mask, rectangle,   
                backgroundModel, foregroundModel, 
                3, cv2.GC_INIT_WITH_RECT) 
    
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 
    
    image = image * mask2[:, :, np.newaxis] 
    url = "D:/fishrecognition/fishrecognition/media/bgremove/"+name
    cv2.imwrite(url, image)

    return url

def bgTransparent(url,name):

    img = Image.open(url)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    url = "D:/fishrecognition/fishrecognition/media/bgTransparent/"+name+'.png'
    img.save(url)

    return url

def resizeImage(url,name):
    img = cv2.imread(url, cv2.IMREAD_UNCHANGED)
    print('Original Dimensions : ',img.shape)
    
    scale_percent = 70
    width = img.shape[1]
    height = img.shape[0]
    while (height>100) and (width>250):
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        scale_percent= scale_percent-1
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Resized Dimensions : ',resized.shape)
    url ="D:/fishrecognition/fishrecognition/media/resizeImage/"+name+".png"
    cv2.imwrite(url, resized)
    
    return url
    

def finalImage(url,raw_name):
        
    # open the image 
    Image1 = Image.open('D:/fishrecognition/fishrecognition/media/auth/bg.png') 
    
    # make a copy the image so that  
    # the original image does not get affected 
    Image1copy = Image1.copy() 
    Image2 = Image.open(url)
    # print('Resized Dimensions : ',int(Image2.shape[1]))
    size = 250, 100
    Image2.thumbnail(size, Image.ANTIALIAS)
    Image2copy = Image2.copy() 

    x = 250 - int(Image2.size[0])
    y = 100- int(Image2.size[1])

    print('Shape : ',Image2.size)
    print('x : ',x)
    print('y : ',y)
    # paste image giving dimensions 
    Image1copy.paste(Image2copy, (int(x/2),int(y/2))) 
    
    # save the image  
    url ="D:/fishrecognition/fishrecognition/media/final/"+raw_name+".png"
    Image1copy.save(url) 
    
    return url

def coba(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['photo']
        fs = FileSystemStorage()
        
        name = fs.save(uploaded_file.name,uploaded_file)
        print(name)
        raw_name = name.split('.')
        raw_name = raw_name[0]
        url = fs.url(name)
        img_path_raw="D:/fishrecognition/fishrecognition"+ str(url)

    prepros = objectDetection(img_path_raw,name)
    removeBg = bgRemover(prepros,name)
    transparentBg = bgTransparent(removeBg,raw_name)
    imgResize = resizeImage(transparentBg,raw_name)
    newGambar = finalImage(imgResize,raw_name)
    result = tes(newGambar)

    post = Histories(fishinput=url,fishoutput=newGambar,species=result[0],result=result[1])
    post.save()

    return render(request,'showcase.html',{'result':result,'img':url,'species':result[0],'percentage':result[1]})

def admintes(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['photo']
        fs = FileSystemStorage()
        
        name = fs.save(uploaded_file.name,uploaded_file)
        print(name)
        raw_name = name.split('.')
        raw_name = raw_name[0]
        url = fs.url(name)
        img_path_raw="D:/fishrecognition/fishrecognition"+ str(url)

    prepros = objectDetection(img_path_raw,name)
    removeBg = bgRemover(prepros,name)
    transparentBg = bgTransparent(removeBg,raw_name)
    imgResize = resizeImage(transparentBg,raw_name)
    newGambar = finalImage(imgResize,raw_name)
    result = tes(newGambar)
    hasil = result[0]+" : "+str(result[1])+"%"
    url_cropped = '/media/cropped/'+name
    url_removebg = '/media/bgremove/'+name
    url_transparentbg = '/media/bgTransparent/'+raw_name+'.png'
    url_resize = '/media/resizeImage/'+raw_name+'.png'
    url_final = '/media/final/'+raw_name+'.png'


    post = Histories(fishinput=url,cropped=url_cropped,removebg=url_removebg, transparentbg=url_transparentbg,resize=url_resize, fishoutput=url_final,species=result[0],result=result[1])
    post.save()

    return render(request,'admin.html',{'hasil':hasil,'img':url,'species':result[0],'percentage':result[1],'cropped':url_cropped,'final':newGambar})


def dataset(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['photo']
        fs = FileSystemStorage()
        
        name = fs.save(uploaded_file.name,uploaded_file)
        print(name)
        raw_name = name.split('.')
        raw_name = raw_name[0]
        url = fs.url(name)
        img_path_raw="D:/fishrecognition/fishrecognition"+ str(url)

    img = cv2.imread(img_path_raw)
    #rgb
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_url = "D:/fishrecognition/fishrecognition/media/datasets/rgb_"+raw_name+".png"
    cv2.imwrite(rgb_url,rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # Canny
    canny = cv2.Canny(gray, 50, 50)
    canny = cv2.bitwise_not(canny)
    canny_url = "D:/fishrecognition/fishrecognition/media/datasets/canny_"+raw_name+".png"
    cv2.imwrite(canny_url,canny)
    #blending image
    canny = cv2.imread(canny_url)
    blending = cv2.addWeighted(canny, 0.2, rgb, 0.6, 0)
    blending_url = "D:/fishrecognition/fishrecognition/media/datasets/blending_"+raw_name+".png"
    cv2.imwrite(blending_url,blending)

    obj = Datasets()
    obj.fishinput = str(url)
    obj.rgb = "/media/datasets/rgb_"+raw_name+".png"
    obj.canny ="/media/datasets/canny_"+raw_name+".png"
    obj.blending= "/media/datasets/blending_"+raw_name+".png"
    obj.save()

    obj = Datasets.objects.latest('id')
    context ={
        'id':obj.id,
        'fishinput':obj.fishinput,
        'rgb':obj.rgb,
        'canny':obj.canny,
        'blending':obj.blending,
    }

    return render(request,'datasets.html',context)

def preprocessing(request):
    obj=Histories.objects.latest('id')
    context ={
        'id':obj.id,
        'fishinput':obj.fishinput,
        'cropped':obj.cropped,
        'removebg':obj.removebg,
        'transparentbg':obj.transparentbg,
        'resize':obj.resize,
        'fishoutput':obj.fishoutput
    }

    return render(request,'preprocessing.html',context)

def tes(url):
    
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

    img_path=url
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
        return ('unknown',0)
    else:
        return (species,percentage)

def convo(request):
    if request.method == 'POST':
        image_path = request.POST['image_path']
        # index = int(request.POST['layer'])

        trained_model = load_model('D:/FIXTA/FishRecognitionVGG19data50.h5')
        img = image.load_img(image_path, target_size=(100, 250))
        test_image = image.img_to_array(img)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image /= 255.

        layer_outputs = [layer.output for layer in trained_model.layers[:17] if not layer.name.startswith('input')]
        # Extracts the outputs of the top 12 layers
        activation_model = models.Model(inputs=trained_model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
        activations = activation_model.predict(test_image) 

        layer_names = []
        for layer in trained_model.layers[0:17]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
            
        images_per_row = 8
        for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
            n_features = layer_activation.shape[-1] # Number of features in the feature map
        #     print(n_features)
            height = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
            width = layer_activation.shape[2]
        #     print(str(height)+','+str(width))
            n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
            display_grid = np.zeros((height * n_cols, images_per_row * width))
            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,:, :,col * images_per_row + row]
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        #             size = (col + 1) * height
        #             print(size)
                    display_grid[col * height : (col + 1) * height, row * width : (row + 1) * width] = channel_image
            scale = 1. / height
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.savefig('D:/fishrecognition/fishrecognition/media/convo/'+layer_name+'.png')
    
        return render(request,'convo.html')
    else:
        return render(request,'convo.html')


class HistoriesView(viewsets.ModelViewSet):
    queryset = Histories.objects.all()
    serializer_class = HistoriesSerializer

class FishesView(viewsets.ModelViewSet):
    queryset = Fishes.objects.all()
    serializer_class = FishesSerializer
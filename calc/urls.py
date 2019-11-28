from django.urls import path, include
from  . import views
from django.conf import settings
from django.conf.urls.static import static
from rest_framework import routers

router = routers.DefaultRouter()
router.register('/histories',views.HistoriesView),
router.register('/fishes',views.FishesView)

urlpatterns = [
    path('api', include(router.urls)),
    path('',views.home, name='home'),
    path('admindash',views.admindash, name='admindash'),
    path('setting',views.setting, name='setting'),
    path('training',views.training, name='training'),
    path('datasets',views.datasets, name='datasets'),
    path('showcase',views.showcase, name='showcase'),
    path('preprocessing',views.preprocessing,name='preprocessing'),
    path('convo',views.convo,name='convo'),


    path('add',views.add, name='add'),
    path('dataset',views.dataset,name='dataset'),
    path('coba',views.coba,name='coba'),
    path('admintes',views.admintes,name='admintes'),
    path('upload',views.upload,name='upload'),
]

if settings.DEBUG:
    urlpatterns +=static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
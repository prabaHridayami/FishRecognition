from django.urls import path, include
from  . import views
from django.conf import settings
from django.conf.urls.static import static
from rest_framework import routers

router = routers.DefaultRouter()
# router.register('/histories',views.HistoriesView),
# router.register('/fishes',views.FishesView)

urlpatterns = [
    path('api', include(router.urls)),
    path('',views.home, name='home'),
    path('admindash',views.admindash, name='admindash'),
    path('datatables',views.datatables, name='datatables'),
    path('training',views.training, name='training'),
    path('datasets',views.datasets, name='datasets'),
    path('showcase',views.showcase, name='showcase'),

    path('add',views.add, name='add'),
    # path('tes',views.tes,name='tes'),
    path('coba',views.coba,name='coba'),
    path('upload',views.upload,name='upload')
]

if settings.DEBUG:
    urlpatterns +=static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
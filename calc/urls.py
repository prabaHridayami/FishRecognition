from django.urls import path
from  . import views

urlpatterns = [
    path('',views.home, name='home'),
    path('showcase',views.showcase, name='showcase'),

    path('add',views.add, name='add')
]
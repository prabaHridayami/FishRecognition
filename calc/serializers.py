from rest_framework import serializers
from .models import Histories, Fishes

class HistoriesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Histories
        fields = ('id','fishinput','fishoutput','species','result','created_at')

class FishesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Fishes
        fields = ('id','fishspecies','created_at')
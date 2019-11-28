from django.db import models
from django.utils import timezone

# Create your models here.

class Fishes(models.Model):
    fishspecies = models.CharField(max_length=200)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.fishspecies

class Histories(models.Model):
    fishinput = models.CharField(max_length=200)
    cropped = models.CharField(max_length=200)
    removebg = models.CharField(max_length=200)
    transparentbg = models.CharField(max_length=200)
    resize = models.CharField(max_length=200)
    fishoutput = models.CharField(max_length=200)
    species = models.CharField(max_length=200)
    result = models.DecimalField(max_digits=5, decimal_places=2)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.id

class Datasets(models.Model):
    fishinput = models.CharField(max_length=200)
    rgb = models.CharField(max_length=200)
    canny = models.CharField(max_length=200)
    blending = models.CharField(max_length=200)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.id



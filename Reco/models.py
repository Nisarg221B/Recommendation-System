from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
import re
# Create your models here.

class RecoUser(models.Model):
    RUser = models.OneToOneField(User, on_delete=models.CASCADE)
    SEX_CHOICES = (
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    )
    DIET_CHOICES = (
        ('Veg', 'Veg'),
        ('Non-Veg', 'Non Veg'),
        ('N/P', 'No Preference'),
    )
    STATE_CHOICES= (
        ('Andhra Pradesh','Andhra Pradesh')	,
        ('Arunachal Pradesh','Arunachal Pradesh'),	
        ('Assam','Assam'),		
        ('Bihar','Bihar'),	
        ('Chhattisgarh','Chhattisgarh'),	
        ('Goa','Goa'),	
        ('Gujarat','Gujarat'),	
        ('Haryana','Haryana'),	
        ('Himachal','Himachal'), 
        ('Jharkhand','Jharkhand'),	
        ('Karnataka','Karnataka'),	
        ('Kerala','Kerala'),
        ('Madhya Pradesh','Madhya Pradesh'), 
        ('Maharashtra','Maharashtra'),	
        ('Manipur','Manipur'),	
        ('Meghalaya','Meghalaya'),	
        ('Mizoram','Mizoram'),	
        ('Nagaland','Nagaland'),	
        ('Odisha','Odisha'),	
        ('Punjab','Punjab'),	
        ('Rajasthan','Rajasthan'),	
        ('Sikkim','Sikkim'),		
        ('Tamil Nadu','Tamil Nadu'),		
        ('Telangana','Telangana'),	
        ('Tripura','Tripura'),
        ('Uttar Pradesh','Uttar Pradesh'),	
        ('Uttarakhand','Uttarakhand'),		
        ('West Bengal','West Bengal'),
        ('N/P', 'No Preference'),
    )
    REGION_CHOICES = (
        ('North', 'North'),
        ('South', 'South'),
        ('East', 'East'),
        ('West', 'West'),
        ('North-East', 'North-East'),
        ('Indo-chinese', 'Indo-chinese'),
        ('Western', 'Western'),
        ('N/P', 'No Preference'),
    )
    FLAVOUR_CHOICES = (
        ('Sweet', 'Sweet'),
        ('Spicy', 'Spicy'),
        ('Sour', 'Sour'),
        ('Bitter', 'Bitter'),
        ('N/P', 'No Preference'),
    )

    name = models.CharField(max_length=32, help_text="Enter your name",blank=False)
    age = models.PositiveIntegerField(default=18, help_text="Enter your age")
    sex = models.CharField(max_length=1, choices=SEX_CHOICES, default='M',help_text="Sex")
    address = models.CharField(max_length=128, help_text="Enter your address")
    phone = models.CharField(max_length=10,help_text="Enter your mobile number of 10 digits")
    diet = models.CharField(max_length=32, choices=DIET_CHOICES, default='N/P',help_text="Select type of Diet")
    region = models.CharField(max_length=32, choices=REGION_CHOICES, default='N/P',help_text="Select your region")
    state = models.CharField(max_length=32, choices=STATE_CHOICES, default='N/P',help_text="Select your state")
    flavour = models.CharField(max_length=32, choices=FLAVOUR_CHOICES, default='N/P',help_text="Which flavour do you prefer?")
    ingredient=models.CharField(max_length=128, help_text="Enter your ingredient preferences separated by commas",default="")
    positiveFeature=models.JSONField(default=list)
    negativeFeature=models.JSONField(default=list)
    features=models.JSONField(default=dict)
    recentfeature = models.JSONField(default=list)
    pastRatings = models.JSONField(default=list)
    satList = models.JSONField(default=list)
    class Meta:
        permissions=(
        )
    def __str__(self):
        return self.RUser.username


class Restaurant(models.Model):
    restaurantId = models.AutoField(primary_key=True)
    name = models.CharField(max_length=64)
    address = models.CharField(max_length=128)
    cuisine = models.CharField(max_length=64)
    rating = models.FloatField()
    totalRatings = models.CharField(max_length=64)
    def __str__(self):
        return self.name

class menuItem(models.Model):
    itemId = models.AutoField(primary_key=True)
    name = models.TextField()
    description = models.TextField(max_length=512)
    price = models.IntegerField()
    rating = models.FloatField()
    category = models.CharField(max_length=128)
    diet = models.CharField(max_length=64)
    restaurantId = models.ForeignKey(Restaurant, on_delete=models.CASCADE, help_text="Enter ID")
    numRatings = models.IntegerField(default=1)
    link=models.IntegerField(default=-1)
    features=models.JSONField(default=list)
    def __str__(self):
        return self.name
from django import forms
from django.db.models.fields import FloatField, IntegerField
#from django.db import models
from Reco.models import RecoUser,Restaurant,menuItem
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
import re
from Reco.models import RecoUser,Restaurant,menuItem

class userRegisterFormA(forms.ModelForm):
    password=forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    username = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))
    def clean_password(self):
        data=self.cleaned_data['password']
        if len(data)<8:
            raise ValidationError(('Password is too short'))
        special_characters = "['~','!','@','#','$','%','&','*','_',';']"
        if not any(char.isdigit() for char in data):
            raise ValidationError(('Password must contain at least 1 digit'))
        if not any(char.isalpha() for char in data):
            raise ValidationError(('Password must contain at least 1 alphabet'))
        if not any(char in special_characters for char in data):
            raise ValidationError(('Password must contain at least 1 special character'))
        return data

    def clean_username(self):
        data=self.cleaned_data['username']
        reg="^[a-zA-Z0-9_-]*$"
        if re.search(reg, data):
            print("valid")
        else:
            raise ValidationError(('Username can only contain alphanumeric and underscore,hyphen!'))
        return data

    class Meta:
        model=User
        fields=('username','email','password')

class userRegisterFormB(forms.ModelForm):
    CHOICES = (
        ('potato','Potato'),
        ('paneer','Paneer'),
        ('milk','Milk'),
        ('rice','Rice'),
        ('cheese','Cheese'),
        ('sugar','Sugar'),
        ('yoghurt','Yoghurt'),
        ('chocolate','Chocolate'),
        ('ghee','Ghee'),
        ('vinegar','Vinegar'),
        ('coconut','Coconut'),
        ('flour','Flour'),
        ('Rice flour','Rice flour'),
        ('Chana dal','Chana dal'),
        ('tomato','Tomato'),
        ('ginger','Ginger'),
        ('garlic','Garlic'),
        ('onions','Onions'),
        ('garam masala','Garam masala'),
        ('Whole wheat flour','Whole wheat flour'),
        ('urad dal','Urad dal'),
        ('chilli','Chilli'),
        ('capsicum','Capsicum'),
        ('lemon','Lemon'),
        ('carrot','Carrot'),
        ('beans','Beans'),
        ('cabbage','Cabbage'),
        ('chicken','Chicken'),
        ('egg','Egg'),
        ('spring','Spring Onions'),
        ('soya','Soya'),
        ('corn','Corn'),
        ('bell','Bell pepper'),
    )

    multipleIngredients = forms.MultipleChoiceField(choices=CHOICES, widget=forms.CheckboxSelectMultiple(),label='Choose Preferred Ingredients')

    def clean_age(self):
        data=self.cleaned_data['age']
        if data>70:
            raise ValidationError(('You are gonna die soon'))
        if data<18:
            raise ValidationError(('Too young to be a doctor'))
        return data

    def clean_phone(self):
        data=self.cleaned_data['phone']
        reg="^(\d{10})$"
        if len(data)==10 and re.search(reg, data):
            print("valid")
        else:
            raise ValidationError(('Mobile Number must have 10 digits'))
        return data
    
    class Meta():
        model=RecoUser
        fields=('name','age','sex','address','phone','diet','region','state','flavour')

# class ratingForm(forms.Form):
#     rating=FloatField(widget=forms.NumberInput(attrs={'class': 'form-control','min':'1','max':'5','step':'0.1'}))
#     id=IntegerField(widget=forms.HiddenInput())

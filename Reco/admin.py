from django.contrib import admin
from Reco.models import RecoUser,Restaurant,menuItem
# Register your models here.
admin.site.register(RecoUser)
admin.site.register(Restaurant)
admin.site.register(menuItem)
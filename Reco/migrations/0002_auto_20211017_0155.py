# Generated by Django 3.2.7 on 2021-10-16 20:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Reco', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='menuitem',
            name='category',
            field=models.CharField(max_length=64),
        ),
        migrations.AlterField(
            model_name='menuitem',
            name='diet',
            field=models.CharField(max_length=64),
        ),
        migrations.AlterField(
            model_name='menuitem',
            name='name',
            field=models.CharField(max_length=64),
        ),
        migrations.AlterField(
            model_name='restaurant',
            name='cuisine',
            field=models.CharField(max_length=64),
        ),
        migrations.AlterField(
            model_name='restaurant',
            name='name',
            field=models.CharField(max_length=64),
        ),
        migrations.AlterField(
            model_name='restaurant',
            name='totalRatings',
            field=models.CharField(max_length=64),
        ),
    ]

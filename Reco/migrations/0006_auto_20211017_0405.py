# Generated by Django 3.2.7 on 2021-10-16 22:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Reco', '0005_auto_20211017_0233'),
    ]

    operations = [
        migrations.AlterField(
            model_name='menuitem',
            name='description',
            field=models.TextField(max_length=512),
        ),
        migrations.AlterField(
            model_name='menuitem',
            name='name',
            field=models.TextField(),
        ),
    ]
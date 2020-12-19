from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

# Create your models here.
class Post(models.Model):
    content = models.TextField(blank=False, default='')
    num_beam = models.IntegerField(
        blank=False, 
        default=1, 
        validators=[
            MaxValueValidator(10), 
            MinValueValidator(1)
        ]
    )

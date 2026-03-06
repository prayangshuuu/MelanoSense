from django.db import models
from django.contrib.auth.models import User
import uuid
import os

class MedicalImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_file = models.ImageField(upload_to='medical_images/originals/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Original {self.id} uploaded at {self.uploaded_at}"

class CroppedImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    parent_image = models.ForeignKey(MedicalImage, on_delete=models.CASCADE, related_name='crops')
    cropped_file = models.ImageField(upload_to='medical_images/crops/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Cropped ROI from {self.parent_image.id} created at {self.created_at}"

class Scan(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='scans')
    image = models.ForeignKey(MedicalImage, on_delete=models.CASCADE)
    risk_level = models.CharField(max_length=20, choices=[('Low', 'Low Risk'), ('Moderate', 'Moderate Risk'), ('High', 'High Risk')])
    confidence = models.FloatField()
    age = models.IntegerField()
    sex = models.CharField(max_length=10)
    localization = models.CharField(max_length=50)
    heatmap_image = models.ImageField(upload_to='analysis/heatmaps/', null=True, blank=True)
    roi_image = models.ImageField(upload_to='analysis/roi/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Scan {self.id} - {self.risk_level} ({self.confidence}%)"

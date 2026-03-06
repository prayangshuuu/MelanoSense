from django.db import models
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

from django import forms
from django.core.exceptions import ImproperlyConfigured

from .utils import label_encoders


def _get_localization_classes():
    """
    Return the list of localization classes directly from the trained
    LabelEncoder so that the form stays in sync with the model.
    """
    encoder = label_encoders.get('localization')
    if encoder is None or not hasattr(encoder, 'classes_'):
        raise ImproperlyConfigured(
            "Localization label encoder is not loaded. "
            "Ensure 'meta_label_encoders.pkl' contains a 'localization' encoder "
            "and that 'load_models()' has been called at startup."
        )
    # Ensure we always work with plain Python strings
    return [str(c) for c in encoder.classes_]


def _build_localization_choices():
    """
    Build (value, label) choices for the localization field.
    Internal value is the exact encoder class string.
    Display label is a human-friendly version.
    """
    classes = _get_localization_classes()
    choices = []
    for c in classes:
        # Simple human-friendly label: title-case and replace underscores if any
        label = c.replace('_', ' ').title()
        choices.append((c, label))
    return choices


class PredictionForm(forms.Form):
    # Choices for 'sex' field, matching typical datasets
    SEX_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
    ]

    image = forms.ImageField(
        label='Upload Skin Lesion Image',
        required=True,
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    
    age = forms.IntegerField(
        label='Age',
        required=True,
        min_value=1,
        max_value=120,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 45'})
    )
    
    sex = forms.ChoiceField(
        label='Sex',
        choices=SEX_CHOICES,
        required=True,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Choices are populated dynamically from the encoder in __init__
    localization = forms.ChoiceField(
        label='Lesion Localization',
        choices=(),
        required=True,
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dynamically bind localization choices from the trained encoder
        self.fields['localization'].choices = _build_localization_choices()

    def clean_localization(self):
        """
        Ensure that the submitted localization value is one of the encoder's
        known classes. This prevents downstream encoding errors.
        """
        value = self.cleaned_data.get('localization')
        classes = _get_localization_classes()
        if value not in classes:
            raise forms.ValidationError(
                "Selected localization is not supported by the prediction model."
            )
        return value

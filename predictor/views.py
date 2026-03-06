import io
import os
import sys
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .forms import PredictionForm
from .utils import hybrid_inference, get_risk_metadata, generate_gradcam_overlay, cnn_model
from .models import MedicalImage, Scan
import base64
from django.http import HttpResponse
from django.template.loader import get_template, render_to_string
from xhtml2pdf import pisa

logger = logging.getLogger(__name__)

def landing(request):
    return render(request, 'landing.html')

def register(request):
    if request.method == 'POST':
        username = request.POST.get('email')
        email = request.POST.get('email')
        password = request.POST.get('password')
        name = request.POST.get('name')
        
        if User.objects.filter(username=username).exists():
            return render(request, 'registration/register.html', {'error': 'Email already registered'})
            
        user = User.objects.create_user(username=username, email=email, password=password)
        user.first_name = name
        user.save()
        
        login(request, user)
        return redirect('dashboard')
    return render(request, 'registration/register.html')

def documentation(request):
    return render(request, 'documentation.html')

@login_required
def dashboard(request):
    recent_scans = Scan.objects.filter(user=request.user).order_by('-created_at')[:5]
    return render(request, 'dashboard/dashboard.html', {
        'recent_scans': recent_scans,
        'user': request.user
    })

@login_required
def history(request):
    risk_filter = request.GET.get('risk')
    scans = Scan.objects.filter(user=request.user).order_by('-created_at')
    
    if risk_filter in ['Low', 'Moderate', 'High']:
        scans = scans.filter(risk_level=risk_filter)
        
    return render(request, 'dashboard/history.html', {
        'scans': scans,
        'selected_risk': risk_filter
    })

def demo_login(request):
    username = 'demo_user'
    email = 'demo@melanosense.ai'
    password = 'demo123'
    
    user, created = User.objects.get_or_create(username=username, email=email)
    if created:
        user.set_password(password)
        user.save()
    
    user = authenticate(username=username, password=password)
    if user is not None:
        login(request, user)
        return redirect('dashboard')
    return redirect('login')

@login_required
def index(request):
    result = None
    image_base64 = None
    error = None
    scan_id = request.GET.get('id')
    
    # Handle historical scan loading via GET id parameter
    if scan_id and request.method == 'GET':
        try:
            scan = Scan.objects.get(id=scan_id, user=request.user)
            risk_meta = get_risk_metadata(scan.confidence)
            
            # Ensure heatmap is generated (Sync logic with scan_result)
            heatmap_url = None
            try:
                heatmap_url = generate_gradcam_overlay(
                    image_path=scan.image.original_file.path,
                    model=cnn_model,
                    scan_id=scan.id
                )
            except Exception as e:
                logger.error(f"Lazy heatmap generation failed in index: {e}")

            # Pass variables directly to context for template parity
            context = {
                'scan': scan,
                'risk': risk_meta,
                'prediction': "CANCER" if scan.confidence >= 40 else "NON-CANCEROUS",
                'heatmap_url': heatmap_url,
                'confidence_formatted': f"{scan.confidence:.2f}",
                'result_view': True # Flag to show result section
            }
            if scan.image and scan.image.original_file:
                with open(scan.image.original_file.path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            context.update({
                'form': form,
                'image_base64': image_base64,
                'error': error
            })
            return render(request, 'index.html', context)

        except Scan.DoesNotExist:
            error = "Scan report not found."
        except Exception as e:
            logger.exception(f"Error loading historical scan: {e}")
            error = "Could not load the requested scan report."

    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Get form data
                age = form.cleaned_data['age']
                sex = form.cleaned_data['sex']
                localization = form.cleaned_data['localization']
                
                # Check for resized image from frontend (base64)
                image_resized_data = request.POST.get('image_resized')
                
                if image_resized_data and 'base64,' in image_resized_data:
                    format, imgstr = image_resized_data.split(';base64,')
                    image_data = base64.b64decode(imgstr)
                    image_base64 = imgstr
                    
                    from io import BytesIO
                    image_file = BytesIO(image_data)
                    
                    # Save to MedicalImage
                    from django.core.files.base import ContentFile
                    med_img = MedicalImage()
                    med_img.original_file.save(f"scan_{request.user.id}.jpg", ContentFile(image_data), save=True)
                else:
                    if 'image' not in request.FILES:
                        raise ValueError("No image provided")
                    
                    image_file = request.FILES['image']
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    image_file.seek(0)
                    
                    med_img = MedicalImage(original_file=image_file)
                    med_img.save()

                # Get hybrid prediction
                result = hybrid_inference(
                    image_file,
                    age=age,
                    sex=sex,
                    localization=localization,
                )
                
                # Save Scan result
                scan = Scan.objects.create(
                    user=request.user,
                    image=med_img,
                    risk_level=result['risk_level'],
                    confidence=result['confidence'],
                    age=age,
                    sex=sex,
                    localization=localization
                )
                
                # Generate AI analysis images (Heatmap, ROI)
                try:
                    # New standardized Grad-CAM pipeline
                    image_path = scan.image.original_file.path
                    heatmap_url = generate_gradcam_overlay(
                        image_path=image_path,
                        model=cnn_model,
                        scan_id=scan.id
                    )
                    
                    # Also generate ROI for the report
                    from .utils import save_analysis_images
                    save_analysis_images(scan)
                except Exception as e:
                    logger.error(f"Failed to generate analysis images: {e}")

                # Add ID to result for download button
                result['id'] = str(scan.id)
                
                # Redirect to scan result page
                return redirect('scan_result', scan_id=scan.id)

            except Exception as e:
                logger.exception(f"Prediction failed: {e}")
                error = (
                    "We couldn't complete the analysis for this image and data. "
                    "Please verify that the image is a clear skin lesion photo and that all fields are filled correctly, then try again."
                )
                result = None

    else:
        form = PredictionForm()
    
    return render(request, 'index.html', {
        'form': form,
        'result': result,
        'image_base64': image_base64,
        'error': error,
    })

@login_required
def scan_result(request, scan_id):
    """
    Detailed medical diagnostic report page.
    """
    scan = get_object_or_404(Scan, id=scan_id, user=request.user)
    risk_meta = get_risk_metadata(scan.confidence)
    
    # Ensure heatmap is generated and get URL
    heatmap_url = None
    try:
        heatmap_url = generate_gradcam_overlay(
            image_path=scan.image.original_file.path,
            model=cnn_model,
            scan_id=scan.id
        )
    except Exception as e:
        logger.error(f"Lazy heatmap generation failed: {e}")

    context = {
        'scan': scan,
        'risk': risk_meta,
        'prediction': "CANCER" if scan.confidence >= 40 else "NON-CANCEROUS",
        'heatmap_url': heatmap_url,
        'user': request.user,
        'confidence_formatted': f"{scan.confidence:.2f}"
    }
    return render(request, 'scan_result.html', context)

@login_required
def generate_report_weasyprint(request, scan_id):
    try:
        # macOS Homebrew Workaround: Inject paths before import
        if sys.platform == "darwin":
            brew_lib_path = "/opt/homebrew/lib"
            brew_bin_path = "/opt/homebrew/bin"
            if os.path.exists(brew_lib_path):
                os.environ['PATH'] = f"{brew_bin_path}:{os.environ.get('PATH', '')}"
                os.environ['DYLD_LIBRARY_PATH'] = f"{brew_lib_path}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
                os.environ['LIBRARY_PATH'] = f"{brew_lib_path}:{os.environ.get('LIBRARY_PATH', '')}"
        
        from weasyprint import HTML
    except (ImportError, OSError) as e:
        logger.error(f"WeasyPrint import failed even after workaround: {e}")
        return HttpResponse(
            "<h3>MelanoSense AI - System Dependency Error</h3>"
            "<p>WeasyPrint cannot find required libraries (Pango/Glib) on your macOS environment.</p>"
            "<p>Please ensure you have installed the dependencies:</p>"
            "<pre style='background:#f1f5f9; padding:10px;'>brew install weasyprint</pre>",
            status=500
        )
    
    try:
        scan = Scan.objects.get(id=scan_id, user=request.user)
        
        # Ensure AI images are generated before report
        # We always check/generate Grad-CAM to ensure we have the latest overlay
        if scan.image:
            from .utils import generate_gradcam_overlay, cnn_model
            generate_gradcam_overlay(scan.image.original_file.path, cnn_model, scan.id)
            scan.refresh_from_db()

        risk_meta = get_risk_metadata(scan.confidence)
        
        context = {
            'scan': scan,
            'risk': risk_meta,
            'prediction': "CANCER" if scan.confidence >= 40 else "NON-CANCEROUS",
            'original_path': scan.image.original_file.path if scan.image else None,
            'heatmap_path': scan.heatmap_image.path if scan.heatmap_image and os.path.exists(scan.heatmap_image.path) else None,
            'roi_path': scan.roi_image.path if scan.roi_image and os.path.exists(scan.roi_image.path) else None,
        }
        
        html_string = render_to_string('reports/diagnostic_report.html', context)
        html = HTML(string=html_string, base_url=request.build_absolute_uri())
        
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="Report ID: {scan.id}.pdf"'
        
        html.write_pdf(response)
        return response
    except Scan.DoesNotExist:
        return HttpResponse("Scan not found", status=404)
    except Exception as e:
        logger.exception(f"WeasyPrint PDF generation failed: {e}")
        return HttpResponse(f"Error: {str(e)}", status=500)

def build_diagnostic_pdf(buffer, scan):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.units import inch
    
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    # Header
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, fontSize=20, spaceAfter=10)
    report_title = f"AI Diagnostic Report : Confidence Score {scan.confidence:.2f}%"
    elements.append(Paragraph(report_title, title_style))
    
    id_style = ParagraphStyle('IDStyle', parent=styles['Normal'], alignment=1, fontSize=12, spaceAfter=20, textColor=colors.grey)
    elements.append(Paragraph(f"Report ID: {scan.id}", id_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Patient Data Table
    data = [
        ["Report ID:", str(scan.id)],
        ["Date:", scan.created_at.strftime("%Y-%m-%d %H:%M")],
        ["Age / Sex:", f"{scan.age} / {scan.sex}"],
        ["Localization:", scan.localization]
    ]
    t = Table(data, colWidths=[1.5*inch, 4*inch])
    t.setStyle(TableStyle([
        ('TEXTCOLOR', (0,0), (0,-1), colors.grey),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*inch))
    
    # Prediction Block
    prediction = "CANCER" if scan.confidence >= 40 else "NON-CANCEROUS"
    risk_meta = get_risk_metadata(scan.confidence)
    
    elements.append(Paragraph("AI Interpretive Results", styles['Heading2']))
    res_data = [
        ["Primary Prediction:", prediction],
        ["Risk Level:", risk_meta['risk_display']],
        ["Confidence:", f"{scan.confidence:.2f}%"]
    ]
    res_t = Table(res_data, colWidths=[2*inch, 3.5*inch])
    res_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('BOX', (0,0), (-1,-1), 1, colors.lightgrey),
        ('PADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(res_t)
    elements.append(Spacer(1, 0.4*inch))
    
    # Images
    elements.append(Paragraph("Image Diagnostics", styles['Heading2']))
    img_data = []
    
    row = []
    # Original
    if scan.image:
        row.append(RLImage(scan.image.original_file.path, 1.8*inch, 1.8*inch))
    else:
        row.append(Paragraph("N/A", styles['Normal']))
        
    # Heatmap
    if scan.heatmap_image and os.path.exists(scan.heatmap_image.path):
        row.append(RLImage(scan.heatmap_image.path, 1.8*inch, 1.8*inch))
    else:
        row.append(Paragraph("N/A", styles['Normal']))
        
    # ROI
    if scan.roi_image and os.path.exists(scan.roi_image.path):
        row.append(RLImage(scan.roi_image.path, 1.8*inch, 1.8*inch))
    else:
        row.append(Paragraph("N/A", styles['Normal']))
        
    img_table = Table([row], colWidths=[2*inch, 2*inch, 2*inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(img_table)
    
    labels = Table([["Original Lesion", "AI Heatmap Overlay", "Isolated ROI Focus"]], colWidths=[2*inch, 2*inch, 2*inch])
    labels.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
    ]))
    elements.append(labels)
    
    # Disclaimer
    elements.append(Spacer(1, 0.5*inch))
    disclaimer_style = ParagraphStyle('Disclaimer', fontSize=9, textColor=colors.red, leading=12)
    elements.append(Paragraph("<b>CLINICAL DISCLAIMER:</b> This AI report is for preliminary screening only. High priority should be given to histopathological correlation.", disclaimer_style))
    
    # Footer
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.drawCentredString(A4[0]/2, 0.5*inch, "Team Minus One • © 2026 MelanoSense")
        canvas.restoreState()
        
    doc.build(elements, onFirstPage=footer, onLaterPages=footer)

@login_required
def generate_report_reportlab(request, scan_id):
    try:
        scan = Scan.objects.get(id=scan_id, user=request.user)
        
        # Ensure heatmap is ready
        if scan.image:
            from .utils import generate_gradcam_overlay, cnn_model
            generate_gradcam_overlay(scan.image.original_file.path, cnn_model, scan.id)
            scan.refresh_from_db()
            
        buffer = io.BytesIO()
        build_diagnostic_pdf(buffer, scan)
        buffer.seek(0)
        
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="Report ID: {scan.id}.pdf"'
        return response
    except Scan.DoesNotExist:
        return HttpResponse("Scan not found", status=404)
    except Exception as e:
        logger.exception(f"ReportLab PDF generation failed: {e}")
        return HttpResponse(f"Error: {str(e)}", status=500)

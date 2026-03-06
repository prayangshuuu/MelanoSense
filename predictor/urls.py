from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('scan/', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('history/', views.history, name='history'),
    path('documentation/', views.documentation, name='documentation'),
    path('demo-login/', views.demo_login, name='demo_login'),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('register/', views.register, name='register'),
    path('scan/result/<uuid:scan_id>/', views.scan_result, name='scan_result'),
    path('report/weasyprint/<uuid:scan_id>/', views.generate_report_weasyprint, name='report_weasyprint'),
    path('report/reportlab/<uuid:scan_id>/', views.generate_report_reportlab, name='report_reportlab'),
    path('download-report/<uuid:scan_id>/', views.generate_report_weasyprint, name='download_report'), # Redirect old download to weasyprint
]

from django.conf.urls import url
from Summarizer import views

urlpatterns = [
    url(r'^api/posts$', views.create_post)
]
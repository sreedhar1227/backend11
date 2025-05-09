from django.urls import path
from .api import list_transcripts, start_interview, submit_answer, end_interview

urlpatterns = [
    path('transcripts/', list_transcripts, name='list_transcripts'),
    path('start_interview/', start_interview, name='start_interview'),
    path('submit_answer/', submit_answer, name='submit_answer'),
     path('end_interview/', end_interview, name='end_interview'),
]
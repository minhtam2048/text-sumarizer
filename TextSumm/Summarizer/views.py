from django.shortcuts import render

from django.http.response import JsonResponse
from django.http import QueryDict
from rest_framework.parsers import JSONParser
from rest_framework import status

from Summarizer.models import Post
from Summarizer.serializers import PostSerializer
from Summarizer.predict import predict
from rest_framework.decorators import api_view

# Create your views here.
@api_view(['PUT'])
def create_post(request):
    if request.method == 'PUT':
        post_data = JSONParser().parse(request)
        post_serializer = PostSerializer(data = post_data)
        if post_serializer.is_valid():
            print('Pre----:', post_serializer.data['content'])
            print('Pre num_beam:', post_serializer.data['num_beam'])
            print('---------------------------')
            summarized = predict(post_serializer.data['content'], post_serializer.data['num_beam'])
            print('Predict----:', summarized)
            print('With num_beam:', post_serializer.data['num_beam'])
            print('---------------------------')
            query_dict = QueryDict('', mutable=True)
            query_dict.update(post_serializer.data)
            query_dict['content'] = summarized
            return JsonResponse(query_dict, status = status.HTTP_200_OK, safe=False)
        return JsonResponse(post_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

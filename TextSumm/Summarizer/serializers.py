from rest_framework import serializers
from Summarizer.models import Post

class PostSerializer(serializers.ModelSerializer):

    class Meta:
        model = Post
        fields = ('id', 'content', 'num_beam')
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.exceptions import NotFound
from rest_framework.response import Response

from app.models import Article 
from app.serializers import ArticleSerializer

@method_decorator(csrf_exempt, name='dispatch')
class ArticleAPIView(APIView):
    def get(self, request, id=0):
        if id == 0:
            articles = Article.objects.all()
            articles_serializer = ArticleSerializer(articles, many=True)
            return Response(articles_serializer.data)
        else:
            try:
                article = Article.objects.get(articleId=id)
                article_serializer = ArticleSerializer(article)
                return Response(article_serializer.data)
            except Article.DoesNotExist:
                raise NotFound("Article not found")
    
    def post(self, request):
        article_data = JSONParser().parse(request)
        articles_serializer = ArticleSerializer(data=article_data)
        if articles_serializer.is_valid():
            articles_serializer.save()
            return Response("Added Successfully")
        return Response("Failed to Add", status=400)
    
    def put(self, request, id):
        article_data = JSONParser().parse(request)
        try:
            article = Article.objects.get(pk=id)

            articles_serializer = ArticleSerializer(article, data=article_data)
            if articles_serializer.is_valid():
                articles_serializer.save()
                return Response("Updated Successfully")
            return Response("Failed to Update", status=400)
        except Article.DoesNotExist:
            raise NotFound("Article not found")
    
    def delete(self, request, id):
        try:
            article = Article.objects.get(articleId=id)
            article.delete()
            return Response("Deleted Successfully")
        except Article.DoesNotExist:
            raise NotFound("Article not found")

 
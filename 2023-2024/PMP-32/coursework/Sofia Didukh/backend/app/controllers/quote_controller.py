from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.exceptions import NotFound
from rest_framework.response import Response

from app.models import Quote 
from app.serializers import QuoteSerializer

@method_decorator(csrf_exempt, name='dispatch')
class QuoteAPIView(APIView):
    def get(self, request, id=0):
        if id == 0:
            quotes = Quote.objects.all()
            quotes_serializer = QuoteSerializer(quotes, many=True)
            return Response(quotes_serializer.data)
        else:
            try:
                quote = Quote.objects.get(quoteId=id)
                quote_serializer = QuoteSerializer(quote)
                return Response(quote_serializer.data)
            except Quote.DoesNotExist:
                raise NotFound("Quote not found")
    
    def post(self, request):
        quote_data = JSONParser().parse(request)
        quotes_serializer = QuoteSerializer(data=quote_data)
        if quotes_serializer.is_valid():
            quotes_serializer.save()
            return Response("Added Successfully")
        return Response("Failed to Add", status=400)
    
    def put(self, request, id):
        quote_data = JSONParser().parse(request)
        try:
            quote = Quote.objects.get(pk=id)

            quotes_serializer = QuoteSerializer(quote, data=quote_data)
            if quotes_serializer.is_valid():
                quotes_serializer.save()
                return Response("Updated Successfully")
            return Response("Failed to Update", status=400)
        except Quote.DoesNotExist:
            raise NotFound("Article not found")
    
    def delete(self, request, id):
        try:
            quote = Quote.objects.get(quoteId=id)
            quote.delete()
            return Response("Deleted Successfully")
        except Quote.DoesNotExist:
            raise NotFound("Article not found")
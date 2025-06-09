from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.exceptions import NotFound
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import ParseError
from django.http import HttpResponse


from app.models import User 
from app.serializers import UserSerializer

@method_decorator(csrf_exempt, name='dispatch')
class UserAPIView(APIView):
    def get(self, request, id=0):
        if  id == 0:
            users = User.objects.all()
            users_serializer = UserSerializer(users, many=True)
            return Response(users_serializer.data)
        else:
            try:
                user = User.objects.get(userId=id)
                user_serializer = UserSerializer(user)
                return Response(user_serializer.data)
            except User.DoesNotExist:
                raise NotFound("Article not found")
    
    def post(self, request):
        user_data = JSONParser().parse(request)
        users_serializer = UserSerializer(data=user_data)
        if users_serializer.is_valid():
            users_serializer.save()
            return Response("Added Successfully")
        return Response("Failed to Add", status=400)
    
    def put(self, request, id):
        try:
            user = User.objects.get(userId=id)
        except User.DoesNotExist:
            raise NotFound("User not found")

        partial_data = request.data

        users_serializer = UserSerializer(user, data=partial_data, partial=True)

        if users_serializer.is_valid():
            users_serializer.save()
            return Response("Updated Successfully")
        return Response(users_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    
    def delete(self, request, id):
        try:
            user = User.objects.get(userId=id)
            user.delete()
            return Response("Deleted Successfully")
        except User.DoesNotExist:
            raise NotFound("Article not found")
        

class IncreaseDaysAPIView(APIView):
    def put(self, request, id):
        request_data = request.data
        try:
            user = User.objects.get(userId=id)
        except User.DoesNotExist:
            return HttpResponse('User not found', status=404)

        payload = {
            'progressDays': user.progressDays + 1
        }
        serializer = UserSerializer(user, data=payload, partial=True)

        if serializer.is_valid():
            serializer.save()
            saved_cigarettes = user.calculate_saved_cigarettes()
            saved_money = user.calculate_saved_money()

            return Response({
                "saved_cigarettes": saved_cigarettes,
                "saved_money": saved_money
            })
        return HttpResponse('Error in update: {}'.format(serializer.errors))


class SavedDataAPIView(APIView):
    def get(self, request, id):
        try:
            user = User.objects.get(userId=id)
        except User.DoesNotExist:
            return HttpResponse('User not found', status=404)

        saved_cigarettes = user.calculate_saved_cigarettes()
        saved_money = user.calculate_saved_money()

        return Response({
            "saved_cigarettes": saved_cigarettes,
            "saved_money": saved_money
        })
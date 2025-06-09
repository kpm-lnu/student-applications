from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed
import jwt, datetime
from rest_framework import status
from django.http import JsonResponse

from app.models import User
from app.serializers import UserSerializer

class RegisterView(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response({"message": "Registered successfully"})

# class LoginView(APIView):
#     def post(self, request):
#         email = request.data['email']
#         password = request.data['password']

#         user = User.objects.filter(email=email).first()

#         if user is None:
#             raise AuthenticationFailed('User not found!')

#         if not user.check_password(password):
#             raise AuthenticationFailed('Incorrect password!')

#         payload = {
#             'id': user.userId,
#             'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
#             'iat': datetime.datetime.utcnow()
#         }

#         token = jwt.encode(payload, 'secret', algorithm='HS256')


#         responce = Response()

#         responce.set_cookie(key='jwt', value=token, httponly=True, max_age=3600)

#         responce.data = {
#             'jwt': token
#         }

#         return responce
class LoginView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        user = User.objects.filter(email=email).first()

        if user is None:
            raise AuthenticationFailed('User not found!')

        if not user.check_password(password):
            raise AuthenticationFailed('Incorrect password!')

        serializer = UserSerializer(user)
        return Response(serializer.data) 
    
class LogoutView(APIView):
    def post(self, request):
        # response = Response()
        # response.delete_cookie('jwt')
        # response.data = {
        #     'message': 'success'
        # }
        # return response
         return JsonResponse({'message': 'success'})

    
class ForgetPasswordView(APIView):
    def put(self, request):
        email = request.data.get('email')
        new_password = request.data.get('password')

        if not all([email, new_password]):
            return Response({'error': 'Please provide email and password'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Find user by email
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({'error': 'User with this email does not exist'}, status=status.HTTP_404_NOT_FOUND)

        user.set_password(new_password)
        user.save()

        return Response({'message': 'Password has been reset successfully'}, status=status.HTTP_200_OK)
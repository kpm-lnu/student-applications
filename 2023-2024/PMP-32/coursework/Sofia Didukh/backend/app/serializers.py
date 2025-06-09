from rest_framework import serializers
from app.models import User, Role, Article, Quote

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('userId', 'userName', 'email', 'password', 'startAmountOfCigarettes', 'priceOfPack', 'amountCigarettesInPack', 'progressDays', 'role')
        extra_kwargs = {
            'password': {'write_only': True},
        }
    
    def create(self, validated_data):
        password = validated_data.pop('password', None)
        instance = self.Meta.model(**validated_data)
        if password is not None:
            instance.set_password(password)
        instance.save()
        return instance
    
    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        for key, value in validated_data.items():
            setattr(instance, key, value)
        if password is not None:
            instance.set_password(password)
        instance.save()
        return instance

class RoleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Role
        fields = ('role') 

class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ('articleId', 'author', 'title', 'text')

class QuoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Quote
        fields = ('quoteId', 'text')

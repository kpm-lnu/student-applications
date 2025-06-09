from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager


class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.pop('startAmountOfCigarettes', None)
        extra_fields.pop('priceOfPack', None)
        extra_fields.pop('amountCigarettesInPack', None)
        extra_fields.pop('progressDays', None)
        extra_fields.setdefault('role_id', 2)


        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(email, password, **extra_fields)

class Role(models.Model):
    role = models.TextField()

class User(AbstractUser):
    userId = models.AutoField(primary_key=True)
    userName = models.TextField()
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)
    startAmountOfCigarettes = models.FloatField(blank=True, null=True)
    priceOfPack = models.FloatField(blank=True, null=True)
    amountCigarettesInPack = models.IntegerField(blank=True, null=True)
    progressDays = models.IntegerField(blank=True, default=0, null=True)
    role = models.ForeignKey(Role, default=1, on_delete=models.CASCADE)
    username = None
    first_name = None
    last_name = None
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    objects = CustomUserManager()


    def calculate_saved_cigarettes(self):
        return self.progressDays * self.startAmountOfCigarettes
    
    def calculate_saved_money(self):
        one_cigarette_price = self.priceOfPack / self.amountCigarettesInPack
        return one_cigarette_price * self.calculate_saved_cigarettes()




class Article(models.Model):
    articleId = models.AutoField(primary_key=True)
    author = models.TextField()
    title = models.TextField()
    text = models.TextField()

class Quote(models.Model):
    quoteId = models.AutoField(primary_key=True)
    text = models.TextField()

from django.urls import path
from .views import article_view, quote_view, register_view, login_view, logout_view, user_view, forger_password_view, increase_days_view, saved_data_view


urlpatterns = [
    path('register/', register_view),
    path('login/', login_view),
    path('logout/', logout_view),
    path('forget-password/', forger_password_view),

    path('user/', user_view),
    path('user/<int:id>/', user_view),
    path('increase/<int:id>/', increase_days_view, name='increase_days'),
    path('saved/<int:id>/', saved_data_view),


    
    path('article/', article_view, name='article-list'),
    path('article/<int:id>/', article_view, name='article-detail'),

    path('quote/', quote_view, name='quote-list'),
    path('quote/<int:id>/', quote_view, name='quote-detail'),    
]
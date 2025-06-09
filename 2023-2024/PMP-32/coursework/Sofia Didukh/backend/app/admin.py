from django.contrib import admin

from .models import User, Article, Role, Quote

admin.site.register(User)
admin.site.register(Article)
admin.site.register(Role)
admin.site.register(Quote)


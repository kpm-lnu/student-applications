from .controllers.user_controller import UserAPIView, IncreaseDaysAPIView, SavedDataAPIView
from .controllers.article_controller import ArticleAPIView
from .controllers.quote_controller import QuoteAPIView
from .controllers.quote_controller import QuoteAPIView
from .controllers.auth_controller import RegisterView, LoginView, LogoutView, ForgetPasswordView





register_view = RegisterView.as_view()
login_view = LoginView.as_view()
logout_view = LogoutView.as_view()
forger_password_view = ForgetPasswordView.as_view()

user_view = UserAPIView.as_view()
article_view =  ArticleAPIView.as_view()
quote_view = QuoteAPIView.as_view()

increase_days_view = IncreaseDaysAPIView.as_view()
saved_data_view = SavedDataAPIView.as_view()

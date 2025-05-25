from common.constants import MAIN_WINDOW_INITIAL_HEIGHT, MAIN_WINDOW_INITIAL_WIDTH


## TODO. Improve class
class ApplicationSettings(object):
    def __init__(self) -> None:
        self.current_width: int = MAIN_WINDOW_INITIAL_WIDTH
        self.current_heigth: int = MAIN_WINDOW_INITIAL_HEIGHT
        self.current_appearance_mode: str = (
            "System"  # Modes: "System" (standard), "Dark", "Light"
        )
        self.current_color_theme: str = (
            "blue"  # Themes: "blue" (standard), "green", "dark-blue"
        )
        self.current_language: str = "English"

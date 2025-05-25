from common.constants import (
    MAIN_WINDOW_MENU_ABOUT_TITLE,
    MAIN_WINDOW_MENU_DOCUMMENTATION_TITLE,
    MAIN_WINDOW_MENU_EXIT_TITLE,
    MAIN_WINDOW_MENU_SETTINGS_TITLE,
    MENU_ABOUT_TAB_MESSAGE,
    MENU_ABOUT_TAB_TITLE,
    MENU_DOCUMMENTATION_URL,
    MENU_EXIT_TAB_MESSAGE,
    MENU_EXIT_TAB_TITLE,
)
from CTkMenuBar import *
from tkinter import messagebox
import webbrowser
from gui.components.window.settings_window import SettingsWindow


class MainWindowMenuBar(CTkMenuBar):
    def __init__(self: CTkMenuBar, parent) -> None:
        self.settings_window = None

        def exit_button_click() -> None:
            messagebox_response = messagebox.askyesno(
                MENU_EXIT_TAB_TITLE, MENU_EXIT_TAB_MESSAGE
            )

            if messagebox_response:
                exit()

        def settings_button_click() -> None:
            if self.settings_window is None or not self.settings_window.winfo_exists():
                self.settings_window = SettingsWindow(parent)
            else:
                self.settings_window.focus()

        def about_button_click() -> None:
            messagebox.showinfo(MENU_ABOUT_TAB_TITLE, MENU_ABOUT_TAB_MESSAGE)

        def docummentation_button_click() -> None:
            webbrowser.open(MENU_DOCUMMENTATION_URL)

        super().__init__(parent)
        self.add_cascade(MAIN_WINDOW_MENU_EXIT_TITLE, exit_button_click)
        self.add_cascade(MAIN_WINDOW_MENU_SETTINGS_TITLE, settings_button_click)
        self.add_cascade(MAIN_WINDOW_MENU_ABOUT_TITLE, about_button_click)
        self.add_cascade(
            MAIN_WINDOW_MENU_DOCUMMENTATION_TITLE, docummentation_button_click
        )

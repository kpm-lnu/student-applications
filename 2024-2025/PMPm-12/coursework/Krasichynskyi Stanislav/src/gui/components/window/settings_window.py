import customtkinter as ctk

from common.constants import (
    MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
    SETTINGS_APPEARANCE_MODE_LABEL,
    SETTINGS_APPEARANCE_MODE_VALUES,
    SETTINGS_CANCEL_BUTTON_LABEL,
    SETTINGS_LANGUAGE_LABEL,
    SETTINGS_LANGUAGE_VALUES,
    SETTINGS_SAVE_BUTTON_LABEL,
    SETTINGS_WINDOW_INITIAL_HEIGHT,
    SETTINGS_WINDOW_INITIAL_WIDTH,
    SETTINGS_WINDOW_TITLE,
)
from gui.components.tab_view.generic_tab_button import GenericSettingsButton
from gui.helpers.window_geometry_helper import center_window_to_display


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self: ctk.CTkToplevel, parent) -> None:
        super().__init__()

        def save_settings_button_click() -> None:
            ### TODO. Add Localization.
            selected_appearance = self.appearance_mode_combobox.get()

            if (
                parent.application_settings.current_appearance_mode
                != selected_appearance
            ):
                parent.application_settings.current_appearance_mode = (
                    selected_appearance
                )
                ctk.set_appearance_mode(
                    parent.application_settings.current_appearance_mode
                )

        def cancel_save_settings_button_click() -> None:
            self.destroy()

        self.title(SETTINGS_WINDOW_TITLE)
        self.geometry(
            center_window_to_display(
                self, SETTINGS_WINDOW_INITIAL_WIDTH, SETTINGS_WINDOW_INITIAL_HEIGHT
            )
        )
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self.appearance_mode_label = ctk.CTkLabel(
            self,
            text=SETTINGS_APPEARANCE_MODE_LABEL,
            width=50,
            font=("Helvetica", 20),
        )
        self.appearance_mode_label.pack(anchor="nw", padx=10, pady=10)

        self.appearance_mode_combobox = ctk.CTkComboBox(
            self,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            height=50,
            corner_radius=20,
            values=SETTINGS_APPEARANCE_MODE_VALUES,
            state="readonly",
            font=("Helvetica", 16),
            dropdown_font=("Helvetica", 16),
        )
        self.appearance_mode_combobox.set(
            parent.application_settings.current_appearance_mode
        )

        self.appearance_mode_combobox.pack(anchor="nw", padx=10, pady=10)

        self.language_label = ctk.CTkLabel(
            self,
            text=SETTINGS_LANGUAGE_LABEL,
            width=50,
            font=("Helvetica", 20),
        )
        self.language_label.pack(anchor="nw", padx=10, pady=10)

        self.language_combobox = ctk.CTkComboBox(
            self,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            height=50,
            corner_radius=20,
            values=SETTINGS_LANGUAGE_VALUES,
            state="readonly",
            font=("Helvetica", 16),
            dropdown_font=("Helvetica", 16),
        )
        self.language_combobox.set(parent.application_settings.current_language)

        self.language_combobox.pack(anchor="nw", padx=10, pady=10)

        self.cancel_button = GenericSettingsButton(
            self,
            SETTINGS_CANCEL_BUTTON_LABEL,
            cancel_save_settings_button_click,
            0.23,
            0.9,
        )
        self.save_button = GenericSettingsButton(
            self,
            SETTINGS_SAVE_BUTTON_LABEL,
            save_settings_button_click,
            0.62,
            0.9,
        )

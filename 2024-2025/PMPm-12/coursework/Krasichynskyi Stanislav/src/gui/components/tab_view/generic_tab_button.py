import customtkinter as ctk

from common.constants import (
    MAIN_WINDOW_TAB_GENERIC_BUTTON_HEIGTH,
    MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
)


class GenericTabButton(ctk.CTkButton):
    def __init__(
        self, parent, text: str, command, relative_x: float, relative_y
    ) -> None:
        super().__init__(
            parent,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            height=MAIN_WINDOW_TAB_GENERIC_BUTTON_HEIGTH,
            text=text,
            command=command,
            font=("Helvetica", 18),
        )

        self.place(relx=relative_x, rely=relative_y)


class GenericSettingsButton(ctk.CTkButton):
    def __init__(
        self, parent, text: str, command, relative_x: float, relative_y
    ) -> None:
        super().__init__(
            parent,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH / 2,
            height=MAIN_WINDOW_TAB_GENERIC_BUTTON_HEIGTH / 1.2,
            text=text,
            command=command,
            font=("Helvetica", 18),
        )

        self.place(relx=relative_x, rely=relative_y)


class WarningGenericTabButton(ctk.CTkButton):
    def __init__(
        self, parent, text: str, command, relative_x: float, relative_y
    ) -> None:
        super().__init__(
            parent,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            height=MAIN_WINDOW_TAB_GENERIC_BUTTON_HEIGTH,
            text=text,
            command=command,
            font=("Helvetica", 18),
            fg_color="#D02222",
            hover_color="#892C2C",
        )

        self.place(relx=relative_x, rely=relative_y)

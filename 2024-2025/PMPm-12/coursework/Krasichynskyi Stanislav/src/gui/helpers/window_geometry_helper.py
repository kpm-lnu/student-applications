import customtkinter as ctk


def center_window_to_display(screen: ctk.CTk, width: int, height: int) -> str:
    """Centers the window to the main display/monitor"""

    screen_width = screen.winfo_screenwidth()
    screen_height = screen.winfo_screenheight()

    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))

    return f"{width}x{height}+{x}+{y}"

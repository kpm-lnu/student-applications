import customtkinter as ctk

from gui.helpers.window_geometry_helper import center_window_to_display


class BoundaryConditionsWindow(ctk.CTkToplevel):
    def __init__(self, parent) -> None: 
        super().__init__()
        
        
        self.title("Boudnary Conditions configuration")
        self.geometry(center_window_to_display(self, 400, 400))
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self.area_types = ctk.CTkLabel(
            self,
            text="CURRENTLY EVERYTHING IS SET TO DIRIHLE CONDITION",
            width=50,
            font=("Helvetica", 12),
        )
        self.area_types.pack(anchor="nw", padx=10, pady=10)

from tkinter import messagebox
import customtkinter as ctk

from common.constants import (
    MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
    PRESETS_CREATE_BUTTON_LABEL,
    PRESETS_GENERAL_LABEL,
    PRESETS_GENERATE_ERROR_MESSAGE,
    PRESETS_POINTS_ON_BOUNDARY_LABEL,
    PRESETS_SHAPE_LABEL,
    PRESETS_VALUES,
)
from gui.components.tab_view.generic_tab_button import GenericSettingsButton
from gui.helpers.area_helper import generate_preset_area
from gui.helpers.window_geometry_helper import center_window_to_display


class PresetWindow(ctk.CTkToplevel):
    def __init__(self, parent) -> None:
        super().__init__()

        def generate_area_button_click() -> None:
            type_of_area = self.presets_combobox.get()
            try:
                number_of_points = int(self.presets_number_of_points.get())

                points, segments = generate_preset_area(type_of_area, number_of_points)

                parent.plot_area.update_area_plot(parent, points)

                parent.area_boundary.points = points
                parent.area_boundary.segments = segments

                if len(parent.area_boundary.points) != 0:
                    parent.tabs.update_area_ready_label()

                self.destroy()

            except ValueError:
                messagebox.showerror(PRESETS_GENERATE_ERROR_MESSAGE)

        self.title(PRESETS_GENERAL_LABEL)
        self.geometry(center_window_to_display(self, 500, 400))
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self.area_types = ctk.CTkLabel(
            self,
            text=PRESETS_SHAPE_LABEL,
            width=50,
            font=("Helvetica", 20),
        )
        self.area_types.pack(anchor="nw", padx=10, pady=10)

        self.presets_combobox = ctk.CTkComboBox(
            self,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            height=50,
            corner_radius=20,
            values=PRESETS_VALUES,
            state="readonly",
            font=("Helvetica", 16),
            dropdown_font=("Helvetica", 16),
        )
        self.presets_combobox.set(PRESETS_VALUES[0])

        self.presets_combobox.pack(anchor="nw", padx=10, pady=10)

        self.number_of_points_label = ctk.CTkLabel(
            self,
            text=PRESETS_POINTS_ON_BOUNDARY_LABEL,
            width=50,
            font=("Helvetica", 20),
        )
        self.number_of_points_label.pack(anchor="nw", padx=10, pady=10)

        self.presets_number_of_points = ctk.CTkEntry(
            self,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            height=50,
            corner_radius=20,
        )
        self.presets_number_of_points.insert(ctk.END, "50")

        self.presets_number_of_points.pack(anchor="nw", padx=10, pady=10)

        self.generate_area = GenericSettingsButton(
            self,
            PRESETS_CREATE_BUTTON_LABEL,
            generate_area_button_click,
            0.5,
            0.8,
        )

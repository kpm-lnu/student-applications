import customtkinter as ctk
from CTkMenuBar import *
from common.constants import (
    MAIN_WINDOW_TITLE,
)
from gui.components.menu.main_window_menu_bar import MainWindowMenuBar
from gui.components.plotter.plot_area import AreaPlotArea
from gui.components.tab_view.main_window_tab_view import MainWindowTabView
from gui.helpers.window_geometry_helper import center_window_to_display
from model.application_settings import ApplicationSettings
from model.area_boundary import AreaBoundary
from model.math_model import MathModel
from model.mesh_data import MeshData
from model.readiness_flags import ReadinessFlags


class MainWindow(ctk.CTk):
    def __init__(self: ctk.CTk, parent, *args, **kwargs) -> None:
        super().__init__()

        ### Variables of class.
        self.application_settings: ApplicationSettings = ApplicationSettings()
        self.area_boundary: AreaBoundary = AreaBoundary()
        self.math_model: MathModel = MathModel()
        self.mesh_object: MeshData = MeshData()
        self.simulation_readiness_flags: ReadinessFlags = ReadinessFlags()

        ### Initial window settings.
        self.title(MAIN_WINDOW_TITLE)
        self.geometry(
            center_window_to_display(
                self,
                self.application_settings.current_width,
                self.application_settings.current_heigth,
            )
        )
        self.resizable(False, False)
        ctk.set_appearance_mode(self.application_settings.current_appearance_mode)
        ctk.set_default_color_theme(self.application_settings.current_color_theme)

        ### Upper menu bar.
        self.menu_bar: MainWindowMenuBar = MainWindowMenuBar(self)

        ### Main control tabs.
        self.tabs: MainWindowTabView = MainWindowTabView(self)

        ### Mesh area.
        self.plot_area: AreaPlotArea = AreaPlotArea(self)

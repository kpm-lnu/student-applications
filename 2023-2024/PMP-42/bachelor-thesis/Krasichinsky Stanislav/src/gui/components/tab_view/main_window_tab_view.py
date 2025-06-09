import os
import customtkinter as ctk
from common.constants import (
    ADHESION_ERROR_MESSAGE,
    APOPTOSIS_ERROR_MESSAGE,
    AREA_TAB_BOUNDARY_CONDITION_BUTTON_TEXT,
    AREA_TAB_CLEAR_AREA_BUTTON_TEXT,
    AREA_TAB_FILE_BUTTON_TEXT,
    AREA_TAB_INPUT_FORMULA_BUTTON_TEXT,
    AREA_TAB_INPUT_FORMULA_BUTTON_TITLE,
    AREA_TAB_MAIN_TITLE,
    AREA_TAB_BOUNDARY_CONDITION_LABEL_TEXT,
    AREA_TAB_MANUAL_INPUT_BUTTON_TEXT,
    AREA_TAB_PRESETS_BUTTON_TEXT,
    AREA_TAB_WRITE_FORMULA_BUTTON_TEXT,
    DEFAULT_ERROR_MESSAGE,
    DIFFUSION_ERROR_MESSAGE,
    MAIN_WINDOW_TAB_AREA_TITLE,
    MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
    MAIN_WINDOW_TAB_MATH_MODEL_TITLE,
    MAIN_WINDOW_TAB_MESH_TITLE,
    MAIN_WINDOW_TAB_START_TITLE,
    MATH_MODEL_TAB_ADHESION_MEASURE_LABEL,
    MATH_MODEL_TAB_APOPTOSIS_MEASURE_LABEL,
    MATH_MODEL_TAB_DIFFUSION_COEFFICIENT_LABEL,
    MATH_MODEL_TAB_GENERAL_LABEL,
    MATH_MODEL_TAB_SET_MODEL_BUTTON_LABEL,
    MESH_TAB_GENERATE_MESH_BUTTON_LABEL,
    MESH_TAB_MAXIMUM_AREA_LABEL,
    MESH_TAB_MESHING_OPTIONS_LABEL,
    MESH_TAB_MESHING_TYPE_LABEL,
    MESH_TAB_MESHING_TYPES,
    MESH_TAB_MINIMUM_ANGLE_LABEL,
    MESH_TAB_NODES_ORDER_LABEL,
    MESH_TAB_NODES_ORDERS_TYPES,
    SET_AREA_ERROR_MESSAGE,
    START_TAB_AREA_SET_LABEL,
    START_TAB_DEFAULT_NOT_READY,
    START_TAB_DEFAULT_READY,
    START_TAB_DEFAUT_VALUE,
    START_TAB_GENERAL_LABEL,
    START_TAB_MESH_SET_LABEL,
    START_TAB_MODEL_SET_LABEL,
    START_TAB_START_BUTTON_LABEL,
)
from gui.components.tab_view.generic_tab_button import (
    GenericTabButton,
    WarningGenericTabButton,
)
from gui.components.window.boundary_conditions_window import BoundaryConditionsWindow
from gui.components.window.manual_input_window import ManualInputWindow
from gui.components.window.presets_window import PresetWindow
from gui.helpers.file_helper import process_file
from gui.helpers.mesh_helper import (
    create_triangulation_options_string,
    triangulate_given_area,
)
from tkinter import messagebox

from model.enums import ValidationStatuses
from simulation_solver.checker import check_math_model_validity
from simulation_solver.main_solver import start_simulation


class MainWindowTabView(ctk.CTkTabview):

    def update_area_ready_label(self) -> None:
        self.is_area_set_value_label.configure(
            text=START_TAB_DEFAULT_READY, fg_color="green"
        )

    def update_mesh_ready_label(self) -> None:
        self.is_mesh_generated_value_label.configure(
            text=START_TAB_DEFAULT_READY, fg_color="green"
        )

    def update_model_ready_label(self) -> None:
        self.is_model_set_value_label.configure(
            text=START_TAB_DEFAULT_READY, fg_color="green"
        )

    def __init__(self: ctk.CTkTabview, parent) -> None:
        super().__init__(
            parent,
            parent.application_settings.current_width - 50,
            parent.application_settings.current_heigth - 50,
            20,
        )
        self.presets_window = None
        self.boundary_condinitons_configuration_window = None
        self.manual_input_window = None

        self.pack()

        def presets_button_click() -> None:
            if self.presets_window is None or not self.presets_window.winfo_exists():
                self.presets_window = PresetWindow(parent)
            else:
                self.presets_window.focus()

        def file_button_click() -> None:
            file_path = ctk.filedialog.askopenfilename(filetypes=[("Images", ".jpg")])
            _, file_extension = os.path.splitext(file_path)

            points, segments = process_file(file_path, file_extension)

            parent.plot_area.update_area_plot(parent, points)

            parent.area_boundary.points = points
            parent.area_boundary.segments = segments

            if len(parent.area_boundary.points) != 0:
                parent.tabs.update_area_ready_label()

        def formula_button_click() -> None:
            dialog = ctk.CTkInputDialog(
                text=AREA_TAB_INPUT_FORMULA_BUTTON_TEXT,
                title=AREA_TAB_INPUT_FORMULA_BUTTON_TITLE,
            )

        def manual_input_button_click() -> None:
            if (
                self.manual_input_window is None
                or not self.manual_input_window.winfo_exists()
            ):
                self.manual_input_window = ManualInputWindow(parent)
            else:
                self.manual_input_window.focus()

        def configure_boundary_conditions_button_click() -> None:
            if (
                self.boundary_condinitons_configuration_window is None
                or not self.boundary_condinitons_configuration_window.winfo_exists()
            ):
                self.presets_window = BoundaryConditionsWindow(parent)
            else:
                self.presets_window.focus()

        def clear_area_button_click() -> None:
            parent.plot_area.clear_area(parent)

        def mesh_area_button_click() -> None:
            if len(parent.area_boundary.points) != 0:
                print("TEST")
                print(parent.area_boundary.points)
                triangulation_options = create_triangulation_options_string(
                    self.mesh_type_combobox.get(),
                    self.nodes_order_combobox.get(),
                    self.minimum_angle.get(),
                    self.maximum_area.get(),
                )

                
                ## TODO. If mesh not triangles why its called triangulate?
                parent.mesh_object = triangulate_given_area(
                    parent.area_boundary.points,
                    parent.area_boundary.segments,
                    triangulation_options,
                    self.nodes_order_combobox.get(),
                )

                parent.plot_area.update_area_triangulation(parent, parent.mesh_object)

                parent.tabs.update_area_ready_label()

            else:
                messagebox.showerror(DEFAULT_ERROR_MESSAGE, SET_AREA_ERROR_MESSAGE)

        def start_simulation_button_click() -> None:
            validation_status: ValidationStatuses = check_math_model_validity(
                parent.math_model.diffusion_coefficient,
                parent.math_model.adhesion_measure,
                parent.math_model.apoptosis_measure,
            )

            if validation_status == ValidationStatuses.OK:
                start_simulation(
                    parent.mesh_object, parent.area_boundary.segments, parent.math_model
                )

            elif validation_status == ValidationStatuses.ADHESION_COEFFICIENT_ERROR:
                messagebox.showerror(DEFAULT_ERROR_MESSAGE, ADHESION_ERROR_MESSAGE)
            elif validation_status == ValidationStatuses.APTOSIS_COEFFICIENT_ERROR:
                messagebox.showerror(DEFAULT_ERROR_MESSAGE, APOPTOSIS_ERROR_MESSAGE)
            elif validation_status == ValidationStatuses.DIFFUSION_COEFFICIENT_ERROR:
                messagebox.showerror(DEFAULT_ERROR_MESSAGE, DIFFUSION_ERROR_MESSAGE)

        def set_model_button_click() -> None:
            try:
                parent.math_model.diffusion_coefficient = float(
                    self.diffusion_coefficient_entry.get()
                )
                parent.math_model.adhesion_measure = float(
                    self.measure_adhesion_entry.get()
                )
                parent.math_model.apoptosis_measure = float(
                    self.measure_apoptosis_entry.get()
                )

                self.update_model_ready_label()

            except ValueError:
                messagebox.showerror("Error", "Incorect value set")

        ### Tabs initialization.
        area_tab = self.add(MAIN_WINDOW_TAB_AREA_TITLE)
        mesh_tab = self.add(MAIN_WINDOW_TAB_MESH_TITLE)
        math_model_tab = self.add(MAIN_WINDOW_TAB_MATH_MODEL_TITLE)
        start_panel_tab = self.add(MAIN_WINDOW_TAB_START_TITLE)

        ### Area tab.
        general_area_label = ctk.CTkLabel(
            area_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 28),
            text=AREA_TAB_MAIN_TITLE,
        )
        general_area_label.place(relx=0, rely=0)

        boundary_condition_label = ctk.CTkLabel(
            area_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 28),
            text=AREA_TAB_BOUNDARY_CONDITION_LABEL_TEXT,
        )
        boundary_condition_label.place(relx=0, rely=0.57)

        presets_button = GenericTabButton(
            area_tab, AREA_TAB_PRESETS_BUTTON_TEXT, presets_button_click, 0, 0.05
        )
        file_button = GenericTabButton(
            area_tab, AREA_TAB_FILE_BUTTON_TEXT, file_button_click, 0, 0.15
        )
        formula_button = GenericTabButton(
            area_tab, AREA_TAB_WRITE_FORMULA_BUTTON_TEXT, formula_button_click, 0, 0.25
        )
        manual_input_button = GenericTabButton(
            area_tab,
            AREA_TAB_MANUAL_INPUT_BUTTON_TEXT,
            manual_input_button_click,
            0,
            0.35,
        )
        boundary_condition_button = GenericTabButton(
            area_tab,
            AREA_TAB_BOUNDARY_CONDITION_BUTTON_TEXT,
            configure_boundary_conditions_button_click,
            0,
            0.62,
        )
        clear_area_button = WarningGenericTabButton(
            area_tab,
            AREA_TAB_CLEAR_AREA_BUTTON_TEXT,
            clear_area_button_click,
            0,
            0.9,
        )

        ### Mesh tab.
        self.minimum_angle = ctk.IntVar(self, 20)
        self.maximum_area = ctk.IntVar(self, 1)

        general_mesh_label = ctk.CTkLabel(
            mesh_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 28),
            text=MESH_TAB_MESHING_OPTIONS_LABEL,
        )
        general_mesh_label.place(relx=0, rely=0)

        mesh_type_label = ctk.CTkLabel(
            mesh_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            text=MESH_TAB_MESHING_TYPE_LABEL,
        )
        mesh_type_label.place(relx=0, rely=0.05)

        self.mesh_type_combobox = ctk.CTkComboBox(
            mesh_tab,
            MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            height=50,
            corner_radius=20,
            values=MESH_TAB_MESHING_TYPES,
            state="readonly",
            font=("Helvetica", 16),
            dropdown_font=("Helvetica", 16),
        )
        self.mesh_type_combobox.place(relx=0, rely=0.11)
        self.mesh_type_combobox.set(MESH_TAB_MESHING_TYPES[0])

        nodes_order_label = ctk.CTkLabel(
            mesh_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            text=MESH_TAB_NODES_ORDER_LABEL,
        )
        nodes_order_label.place(relx=0, rely=0.19)

        self.nodes_order_combobox = ctk.CTkComboBox(
            mesh_tab,
            MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            height=50,
            corner_radius=20,
            values=MESH_TAB_NODES_ORDERS_TYPES,
            state="readonly",
            font=("Helvetica", 16),
            dropdown_font=("Helvetica", 16),
        )
        self.nodes_order_combobox.place(relx=0, rely=0.25)
        self.nodes_order_combobox.set(MESH_TAB_NODES_ORDERS_TYPES[0])

        minimum_angle_label = ctk.CTkLabel(
            mesh_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            text=MESH_TAB_MINIMUM_ANGLE_LABEL,
        )
        minimum_angle_label.place(relx=0, rely=0.34)

        self.minimum_angle_slider = ctk.CTkSlider(
            mesh_tab,
            from_=20,
            to=30,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            variable=self.minimum_angle,
        )
        self.minimum_angle_slider.place(relx=0, rely=0.39)

        maximum_area_label = ctk.CTkLabel(
            mesh_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            text=MESH_TAB_MAXIMUM_AREA_LABEL,
        )
        maximum_area_label.place(relx=0, rely=0.43)

        self.maximum_area_slider = ctk.CTkSlider(
            mesh_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            from_=1,
            to=90,
            variable=self.maximum_area,
        )
        self.maximum_area_slider.place(relx=0, rely=0.49)

        mesh_area_button = GenericTabButton(
            mesh_tab,
            MESH_TAB_GENERATE_MESH_BUTTON_LABEL,
            mesh_area_button_click,
            0,
            0.9,
        )

        ### Mathematical model tab.
        general_math_model_label = ctk.CTkLabel(
            math_model_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 28),
            text=MATH_MODEL_TAB_GENERAL_LABEL,
        )
        general_math_model_label.place(relx=0, rely=0)

        diffusion_coefficient_label = ctk.CTkLabel(
            math_model_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            text=MATH_MODEL_TAB_DIFFUSION_COEFFICIENT_LABEL,
        )
        diffusion_coefficient_label.place(relx=0, rely=0.05)

        self.diffusion_coefficient_entry = ctk.CTkEntry(
            math_model_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            corner_radius=20,
            height=50,
        )
        self.diffusion_coefficient_entry.insert(ctk.END, "1")
        self.diffusion_coefficient_entry.place(relx=0, rely=0.11)

        measure_adhesion_label = ctk.CTkLabel(
            math_model_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            text=MATH_MODEL_TAB_ADHESION_MEASURE_LABEL,
        )
        measure_adhesion_label.place(relx=0, rely=0.19)

        self.measure_adhesion_entry = ctk.CTkEntry(
            math_model_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            corner_radius=20,
            height=50,
        )
        self.measure_adhesion_entry.insert(ctk.END, "1")
        self.measure_adhesion_entry.place(relx=0, rely=0.25)

        measure_apoptosis_label = ctk.CTkLabel(
            math_model_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            text=MATH_MODEL_TAB_APOPTOSIS_MEASURE_LABEL,
        )
        measure_apoptosis_label.place(relx=0, rely=0.34)

        self.measure_apoptosis_entry = ctk.CTkEntry(
            math_model_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            corner_radius=20,
            height=50,
        )
        self.measure_apoptosis_entry.insert(ctk.END, "1")
        self.measure_apoptosis_entry.place(relx=0, rely=0.39)

        set_model_button = GenericTabButton(
            math_model_tab,
            MATH_MODEL_TAB_SET_MODEL_BUTTON_LABEL,
            set_model_button_click,
            0,
            0.9,
        )

        ### Start panel tab.
        general_start_label = ctk.CTkLabel(
            start_panel_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 28),
            text=START_TAB_GENERAL_LABEL,
        )
        general_start_label.place(relx=0, rely=0)

        is_area_set_label = ctk.CTkLabel(
            start_panel_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            text=START_TAB_AREA_SET_LABEL,
        )
        is_area_set_label.place(relx=0, rely=0.05)

        self.is_area_set_value_label = ctk.CTkLabel(
            start_panel_tab,
            width=50,
            font=("Helvetica", 16),
            text=START_TAB_DEFAULT_NOT_READY,
            fg_color="red",
        )
        self.is_area_set_value_label.place(relx=0.25, rely=0.05)

        self.is_mesh_generated_label = ctk.CTkLabel(
            start_panel_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            text=START_TAB_MESH_SET_LABEL,
        )
        self.is_mesh_generated_label.place(relx=0, rely=0.11)

        self.is_mesh_generated_value_label = ctk.CTkLabel(
            start_panel_tab,
            width=60,
            font=("Helvetica", 16),
            text=START_TAB_DEFAULT_NOT_READY,
            fg_color="red",
        )
        self.is_mesh_generated_value_label.place(relx=0.25, rely=0.11)

        is_model_set_label = ctk.CTkLabel(
            start_panel_tab,
            width=MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH,
            font=("Helvetica", 16),
            text=START_TAB_MODEL_SET_LABEL,
        )
        is_model_set_label.place(relx=0, rely=0.19)

        self.is_model_set_value_label = ctk.CTkLabel(
            start_panel_tab,
            width=60,
            font=("Helvetica", 16),
            text=START_TAB_DEFAUT_VALUE,
            fg_color="yellow",
        )
        self.is_model_set_value_label.place(relx=0.25, rely=0.19)

        start_simulation_button = GenericTabButton(
            start_panel_tab,
            START_TAB_START_BUTTON_LABEL,
            start_simulation_button_click,
            0,
            0.9,
        )

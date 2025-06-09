### MAIN WINDOW GUI
MAIN_WINDOW_TITLE: str = "Tumor Simulation Program"
MAIN_WINDOW_INITIAL_WIDTH: int = 1300
MAIN_WINDOW_INITIAL_HEIGHT: int = 880

### MAIN WINDOW UPPER MENU BAR
MAIN_WINDOW_MENU_EXIT_TITLE: str = "Exit"
MAIN_WINDOW_MENU_SETTINGS_TITLE: str = "Settings"
MAIN_WINDOW_MENU_ABOUT_TITLE: str = "About"
MAIN_WINDOW_MENU_DOCUMMENTATION_TITLE: str = "Docummentation"

### MAIN WINDOW MENU CONTROL
MENU_ABOUT_TAB_TITLE: str = "About Program"
MENU_ABOUT_TAB_MESSAGE: str = (
    "Wroten by Stanislaw Krasichynsky \n For Bachelor Dimploma Project \n 2024"
)

MENU_EXIT_TAB_TITLE: str = "Exit Program"
MENU_EXIT_TAB_MESSAGE: str = "Are you sure you want to exit?"

MENU_DOCUMMENTATION_URL: str = (
    "https://github.com/thekrasich/Cancer_Tumor_Simulation_Program"
)

### SETTINGS GUI
SETTINGS_WINDOW_TITLE: str = "Program Settings"
SETTINGS_WINDOW_INITIAL_WIDTH: int = 600
SETTINGS_WINDOW_INITIAL_HEIGHT: int = 680
SETTINGS_APPEARANCE_MODE_LABEL: str = "Appearance Mode"
SETTINGS_APPEARANCE_MODE_VALUES: list[str] = ["System", "Dark", "Light"]
SETTINGS_LANGUAGE_LABEL: str = "Language"
SETTINGS_LANGUAGE_VALUES: list[str] = ["English", "Ukrainian"]
SETTINGS_SAVE_BUTTON_LABEL: str = "Save"
SETTINGS_CANCEL_BUTTON_LABEL: str = "Cancel"


### PLOTTER
PLOTTER_WIDTH_INCHES: float = 7.6
PLOTTER_HEIGTH_INCHES: float = 7.6
PLOTTER_X_PLACEMENT: float = 0.38
PLOTTER_Y_PLACEMENT: float = 0.1

### MAIN WINDOW TABS
MAIN_WINDOW_TAB_AREA_TITLE: str = "Area"
MAIN_WINDOW_TAB_MESH_TITLE: str = "Mesh"
MAIN_WINDOW_TAB_MATH_MODEL_TITLE: str = "Mathematical Model"
MAIN_WINDOW_TAB_START_TITLE: str = "Start Panel"
MAIN_WINDOW_TAB_GENERIC_BUTTON_WIDTH: int = 430
MAIN_WINDOW_TAB_GENERIC_BUTTON_HEIGTH: int = 65

### TABS SECTIONS
AREA_TAB_MAIN_TITLE: str = "Generate the area"
AREA_TAB_BOUNDARY_CONDITION_LABEL_TEXT: str = "Boundary Conditions"
AREA_TAB_PRESETS_BUTTON_TEXT: str = "Choose from Presets"
AREA_TAB_FILE_BUTTON_TEXT: str = "Select file"
AREA_TAB_WRITE_FORMULA_BUTTON_TEXT: str = "Write a formula"
AREA_TAB_INPUT_FORMULA_BUTTON_TEXT: str = "Input a formula"
AREA_TAB_INPUT_FORMULA_BUTTON_TITLE: str = "Formula Input"
AREA_TAB_MANUAL_INPUT_BUTTON_TEXT: str = "Manual points input"
AREA_TAB_BOUNDARY_CONDITION_BUTTON_TEXT: str = "Configure boundary conditions"
AREA_TAB_CLEAR_AREA_BUTTON_TEXT: str = "Clear area"

MESH_TAB_MESHING_OPTIONS_LABEL: str = "Meshing options"
MESH_TAB_MESHING_TYPE_LABEL: str = "Mesh Type"
MESH_TAB_NODES_ORDER_LABEL: str = "Nodes Order"
MESH_TAB_MINIMUM_ANGLE_LABEL: str = "Minimum anlge ( 20 - 30 )"
MESH_TAB_MAXIMUM_AREA_LABEL: str = "Maximum area of element ( 0 - 1 )"
MESH_TAB_GENERATE_MESH_BUTTON_LABEL: str = "Generate mesh"

MESH_TAB_MESHING_TYPES: list[str] = ["Triangular"]
MESH_TAB_NODES_ORDERS_TYPES: list[str] = ["Linear", "Quadratic"]

MATH_MODEL_TAB_GENERAL_LABEL: str = "Mathematical model settings"
MATH_MODEL_TAB_DIFFUSION_COEFFICIENT_LABEL: str = "Diffusion Coefficient (D)"
MATH_MODEL_TAB_ADHESION_MEASURE_LABEL: str = "Measure of adhesion (G)"
MATH_MODEL_TAB_APOPTOSIS_MEASURE_LABEL: str = "Measure of apoptosis (A)"
MATH_MODEL_TAB_SET_MODEL_BUTTON_LABEL: str = "Set Model Values"

START_TAB_GENERAL_LABEL: str = "Start Panel"
START_TAB_START_BUTTON_LABEL: str = "Start"
START_TAB_AREA_SET_LABEL: str = "Area Set: "
START_TAB_MESH_SET_LABEL: str = "Mesh Generated: "
START_TAB_MODEL_SET_LABEL: str = "Model Set: "

START_TAB_DEFAULT_NOT_READY: str = "Not Ready"
START_TAB_DEFAUT_VALUE: str = "Set by Default"
START_TAB_DEFAULT_READY: str = "Ready"

### PRESETS
PRESETS_GENERAL_LABEL: str = "Presets"
PRESETS_SHAPE_LABEL: str = "Shape: "
PRESETS_VALUES: list[str] = ["Donut", "Circle"]
PRESETS_POINTS_ON_BOUNDARY_LABEL: str = "Initial number of points on boundary: "
PRESETS_CREATE_BUTTON_LABEL: str = "Create area"
PRESETS_GENERATE_ERROR_MESSAGE: str = "Error", "Input correct amount of points"

DEFAULT_ERROR_MESSAGE: str = "ERROR"
ADHESION_ERROR_MESSAGE: str = "Set adhesion coefficient correctly"
DIFFUSION_ERROR_MESSAGE: str = "Set diffusion coefficient correctly"
APOPTOSIS_ERROR_MESSAGE: str = "Set aptosis coefficient correctly"
SET_AREA_ERROR_MESSAGE: str = "Set area first!"

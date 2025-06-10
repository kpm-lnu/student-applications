from src.model_solver.aitlen_solver import AitkenSolver
from src.simulation.simulation_service import SimulationService
from src.data.simulation_data import SimulationData
from src.area.reader.area_reader_helper import AreaReaderHelper
from src.area.preset.area_preset_helper import AreaPresetHelper
from src.common.constants import INITIAL_FILE_PATH, TRIANGULATION_OPTIONS


if __name__ == "__main__":
    boundary_points = AreaReaderHelper.read_boundary_points_from_file(INITIAL_FILE_PATH)

    simulationData = SimulationData(TRIANGULATION_OPTIONS, boundary_points)
    
    SimulationService.start_simulation(simulationData)
    
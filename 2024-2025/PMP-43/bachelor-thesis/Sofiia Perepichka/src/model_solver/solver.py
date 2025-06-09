import numpy as np


from src.data.simulation_data import SimulationData
from src.fem.fem_functions import FemService
from src.plotter.plotter_service import PlotterService
from src.common.helpers import HelperFunctions

class SimulationSolver:
    @staticmethod
    def solve_time_dependent( triangulation_results: dict, simulation_data: SimulationData):

        
        stop_time = simulation_data.stop_time
        time_step = simulation_data.time_step
        diffusion_cell_coefficient = simulation_data.diffusion_cell_coefficient
        diffusion_nutrient_coefficient = simulation_data.diffusion_nutrient_coefficient
        proliferation_coefficient = simulation_data.proliferation_coefficient
        necrosis_coefficient = simulation_data.necrosis_coefficient
        nutrient_consumption_coefficient = simulation_data.nutrient_consumption_coefficient
        boundary_transfer_coefficient = simulation_data.boundary_transfer_coefficient
        nutrient_concentration = simulation_data.nutrient_concentration
        nutrient_source = simulation_data.nutrient_source
        initial_cell_density = simulation_data.initial_cell_density
        initial_nutrient_concentration = simulation_data.initial_nutrient_concentration
        
        vertices = triangulation_results["vertices"]
        triangles = triangulation_results["triangles"]
        triangle_vertices = np.array([[vertices[j] for j in i] for i in triangles])

        boundary_edges = HelperFunctions.get_boundary_edges(triangles)
        

        num_nodes = len(vertices)
        C = np.array([initial_cell_density(x, y) for x, y in vertices])
        S = np.array([initial_nutrient_concentration(x, y) for x, y in vertices])

        M_global = np.zeros((num_nodes, num_nodes))
        K_global_C = np.zeros((num_nodes, num_nodes))
        K_global_S = np.zeros((num_nodes, num_nodes))
        M_lambda = np.zeros((num_nodes, num_nodes))
        M_mu = np.zeros((num_nodes, num_nodes))
        M_gamma = np.zeros((num_nodes, num_nodes))
        M_G = np.zeros((num_nodes, num_nodes))

        F_global = np.zeros(num_nodes)
        G_global = np.zeros(num_nodes)


        for i in range(len(triangles)):
            ke_c = np.array(FemService.compute_stiffness_matrix(triangle_vertices[i], a11=diffusion_cell_coefficient, a22=diffusion_cell_coefficient))
            ke_s = np.array(FemService.compute_stiffness_matrix(triangle_vertices[i], a11=diffusion_nutrient_coefficient, a22=diffusion_nutrient_coefficient))
            me = FemService.compute_mass_matrix(triangle_vertices[i])
            me_lambda = proliferation_coefficient * me
            me_mu = necrosis_coefficient * me
            me_gamma = nutrient_consumption_coefficient * me
            qe = FemService.compute_source_vector(triangle_vertices[i], fe=[nutrient_source, nutrient_source, nutrient_source])


            M_global = FemService.compute_global_matrix(M_global, me, triangles[i])
            K_global_C = FemService.compute_global_matrix(K_global_C, ke_c, triangles[i])
            K_global_S= FemService.compute_global_matrix(K_global_S, ke_s, triangles[i])
            M_lambda = FemService.compute_global_matrix(M_lambda, me_lambda, triangles[i])
            M_mu = FemService.compute_global_matrix(M_mu, me_mu, triangles[i])
            M_gamma = FemService.compute_global_matrix(M_gamma, me_gamma, triangles[i])

            F_global = FemService.compute_global_vector(qe, triangles[i], F_global)

            for edge in boundary_edges:
                i, j = edge
                boundary_length = HelperFunctions.compute_boundary_length(vertices[i], vertices[j])
                me_g = FemService.compute_boundary_matrix(boundary_length, beta=1.0, beta_coef= boundary_transfer_coefficient)
                ge = FemService.compute_boundary_vector(boundary_length, psi=nutrient_concentration, beta=1.0, beta_coef= boundary_transfer_coefficient)
                M_G = FemService.compute_global_matrix_boundary(M_G, me_g, edge)
                G_global = FemService.compute_global_vector_boundary(ge, edge, G_global)

        num_time_steps = int(stop_time / time_step) 

        A11 = (1 / time_step) * M_global + 0.5 * K_global_C + 0.5 * M_mu
        A12 = -0.5 * M_lambda
        A21 = 0.5 * M_gamma
        A22 = (1 / time_step) * M_global + 0.5 * K_global_S + 0.5 * M_G

        R11 = (1 / time_step) * M_global - 0.5 * K_global_C - 0.5 * M_mu
        R22 = (1 / time_step) * M_global - 0.5 * K_global_S - 0.5 * M_G


        big_size = 2 * num_nodes
        bigA = np.zeros((big_size, big_size))
        bigA[:num_nodes, :num_nodes] = A11
        bigA[:num_nodes, num_nodes:] = A12
        bigA[num_nodes:, :num_nodes] = A21
        bigA[num_nodes:, num_nodes:] = A22

        for j in range(num_time_steps):
            rhs_C = R11 @ C + 0.5 * M_lambda @ S
            rhs_S = -0.5 * M_gamma @ C + R22 @ S + F_global + G_global

            rhs = np.zeros(big_size)
            rhs[:num_nodes] = rhs_C
            rhs[num_nodes:] = rhs_S


            sol = np.linalg.solve(bigA, rhs)
            C = sol[:num_nodes]
            S = sol[num_nodes:]
            


        
        PlotterService.plot_solution(vertices, C, S, title=f"Final step ")
        return  C, S, M_global, K_global_C, K_global_S


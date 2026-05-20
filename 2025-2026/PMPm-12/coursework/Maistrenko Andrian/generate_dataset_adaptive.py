import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import pandas as pd
import random
import numpy as np
import multiprocessing as mp
import matplotlib
import tempfile
import shutil
matplotlib.use('Agg')  

from FEM.factory import FEMFactory, ElementType
from FEM.material import Material
from FEM.axisymmetricFEMSolver import AxisymmetricFEMSolver
from FEM.postprocessors.fem_postprocessor import FEMPostProcessor, RECOVERY_RAW, RECOVERY_SPR
from FEM.main import ExperimentConfig, create_boundary_conditions
from FEM.FindElementsForRefinement import FindElementsForRefinementIBEM
from FEM.services.io_service import NullIOService

def format_stress_dict(stress_array, prefix=""):
    return {
        f"{prefix}srr": stress_array[0],
        f"{prefix}szz": stress_array[1],
        f"{prefix}srz": stress_array[2],
        f"{prefix}stt": stress_array[3]
    }

def node_id_is_on_boundary(mesh, node_id):
    bnd_nodes = set(mesh.leftBoundaryNodes + mesh.rightBoundaryNodes + 
                    mesh.topBoundaryNodes + mesh.bottomBoundaryNodes)
    return node_id in bnd_nodes

def get_4_neighbors(mesh, node_id):
    target = mesh.nodes[node_id]
    distances = []
    for nid, n in mesh.nodes.items():
        if nid != node_id:
            dist = (n.r - target.r)**2 + (n.z - target.z)**2
            distances.append((dist, nid))
    distances.sort()
    return [nid for dist, nid in distances[:4]]

def process_single_sample(sample_id):
    """Генерація даних для METHOD-B (Адаптивний МСЕ)"""
    
    # 1. Ізоляція потоку у тимчасовій папці (щоб IBEM файли не перетиналися)
    original_cwd = os.getcwd()
    work_dir = f"temp_run_{sample_id}_{os.getpid()}"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    
    local_dataset = []
    
    try:
        r_min, r_max, z_min, z_max = 1.0, 2.0, 0.0, 1.0
        nu, mu_base = 0.3, 0.7
        mat = Material("Steel", E=2 * mu_base * (1 + nu), nu=nu)
        current_p = random.uniform(50.0, 200.0)
        
        print(f"Потік обробляє зразок #{sample_id} (Тиск: {current_p:.1f})")
        
        config = ExperimentConfig(
            r_min=r_min, r_max=r_max, z_min=z_min, z_max=z_max, r_func=lambda z: 0,
            mesh_resolutions=[], p=current_p, nu=nu, mu=mu_base,
            fixed_z=0.5, fixed_r=1, compare_eps=1e-6,
            element_types=[ElementType.LINEAR], n_points_range=[2]
        )
        
        # ==========================================
        # (Initial Mesh)
        # ==========================================
        factory_c = FEMFactory(r_min, r_max, z_min, z_max, 0, 0, material=mat, node_dof=2).init(ElementType.LINEAR)
        _, mesh_c = factory_c.create(n_points=2)
        mesh_c.generate(config.r_min, config.r_max, config.z_min, config.z_max, rN=4, zN=4, r_func=config.r_func)
        
        bcs_c0 = create_boundary_conditions(mesh_c, current_p)
        solver_c0 = AxisymmetricFEMSolver(mesh_c, bcs_c0)
        solver_c0.run(custom_n_points=2, element_type=ElementType.LINEAR)
        
        pp_raw_c0 = FEMPostProcessor(mesh_c, recovery_mode=RECOVERY_RAW)
        
        # зберігаємо стан до згущення
        nodes_info = []
        for target_node_id, node in list(mesh_c.nodes.items()):
            if node_id_is_on_boundary(mesh_c, target_node_id): continue
            neighbors = get_4_neighbors(mesh_c, target_node_id)
            if len(neighbors) != 4: continue
            
            target_pt = np.array([[node.r, node.z]])
            S0_T = pp_raw_c0.stresses_at(target_pt)[0]
            
            neighbors_info = []
            for n_id in neighbors:
                n_node = mesh_c.nodes[n_id]
                n_pt = np.array([[n_node.r, n_node.z]])
                S0_N = pp_raw_c0.stresses_at(n_pt)[0]
                neighbors_info.append({'r': n_node.r, 'z': n_node.z, 'S0_N': S0_N})
                
            nodes_info.append({
                'r': node.r, 'z': node.z,
                'S0_T': S0_T,
                'neighbors': neighbors_info
            })

        # ==========================================
        # (Adaptive Step 1)
        # ==========================================
        ibem_strategy = FindElementsForRefinementIBEM(io=NullIOService())
        refine_ids = ibem_strategy.find(mesh_c, threshold=0.3, bcs=bcs_c0, mode='eta')
        
        if refine_ids:
            mesh_c.refine_elements(refine_ids, auto_plot=False)
            
            bcs_c1 = create_boundary_conditions(mesh_c, current_p)
            solver_c1 = AxisymmetricFEMSolver(mesh_c, bcs_c1)
            mesh_c.build_and_attach_mortar_interfaces(solver_c1)
            solver_c1.run(custom_n_points=2, element_type=ElementType.LINEAR)
            
        pp_raw_c1 = FEMPostProcessor(mesh_c, recovery_mode=RECOVERY_RAW)

        # ==========================================
        # ДРІБНА СІТКА 
        # ==========================================
        factory_f = FEMFactory(r_min, r_max, z_min, z_max, 0, 0, material=mat, node_dof=2).init(ElementType.LINEAR)
        _, mesh_f = factory_f.create(n_points=2)
        mesh_f.generate(config.r_min, config.r_max, config.z_min, config.z_max, rN=32, zN=32, r_func=config.r_func)
        bcs_f = create_boundary_conditions(mesh_f, current_p)
        solver_f = AxisymmetricFEMSolver(mesh_f, bcs_f)
        solver_f.run(custom_n_points=2, element_type=ElementType.LINEAR)
        
        pp_f = FEMPostProcessor(mesh_f, recovery_mode=RECOVERY_SPR)

        # ==========================================
        # ФОРМУВАННЯ ДАТАСЕТУ (Method-B)
        # ==========================================
        for info in nodes_info:
            target_pt = np.array([[info['r'], info['z']]])
            
            S1_T = pp_raw_c1.stresses_at(target_pt)[0]
            F_T = pp_f.stresses_at(target_pt)[0]
            
            row_data = {'sample_id': sample_id, 'r': info['r'], 'z': info['z'], 'p': current_p}
            
            row_data.update(format_stress_dict(info['S0_T'] - S1_T, prefix="PT_diff_"))
            
            for idx, n_info in enumerate(info['neighbors']):
                n_pt = np.array([[n_info['r'], n_info['z']]])
                S1_N = pp_raw_c1.stresses_at(n_pt)[0]
                
                # Вхід А: Різниця (Цикл 0 - Цикл 1) для сусіда
                row_data.update(format_stress_dict(n_info['S0_N'] - S1_N, prefix=f"PN{idx+1}_diff_"))
                # Вхід Б: Локальна варіація на грубій сітці (Сусід_0 - Ціль_0)
                row_data.update(format_stress_dict(n_info['S0_N'] - info['S0_T'], prefix=f"PN{idx+1}_var_"))

            # Ціль (Y): Реальна похибка початкової грубої сітки (Ідеал - Цикл 0)
            row_data.update(format_stress_dict(F_T - info['S0_T'], prefix="TARGET_"))
            local_dataset.append(row_data)
            
    except Exception as e:
        print(f"Помилка у потоці {sample_id}: {e}")
        
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(work_dir, ignore_errors=True)
        
    return local_dataset

def generate_dataset_parallel(num_samples=2000):
    dataset = []
    num_cores = mp.cpu_count()
    print(f"Запуск генерації METHOD-B (Адаптивний) на {num_cores} ядрах...")
    
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(process_single_sample, range(num_samples))
        
    for res in results:
        dataset.extend(res)

    df = pd.DataFrame(dataset)
    filename = "fem_stress_dataset_method_b_2000.csv"
    df.to_csv(filename, index=False)
    print(f"\nГотово! Датасет (Method-B) збережено у '{filename}'. Зібрано {len(dataset)} рядків.")

if __name__ == "__main__":
    start=time.time()
    generate_dataset_parallel(num_samples=2000)
    end=time.time()
    print(f"Took {end-start} seconds")

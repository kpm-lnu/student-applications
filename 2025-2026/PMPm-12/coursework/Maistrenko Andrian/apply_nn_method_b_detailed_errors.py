import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import shutil

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from FEM.factory import FEMFactory, ElementType
from FEM.material import Material
from FEM.axisymmetricFEMSolver import AxisymmetricFEMSolver
from FEM.postprocessors.fem_postprocessor import FEMPostProcessor, RECOVERY_RAW, RECOVERY_SPR
from FEM.main import ExperimentConfig, create_boundary_conditions
from FEM.FindElementsForRefinement import FindElementsForRefinementIBEM
from FEM.services.io_service import NullIOService

directory_name = "results"
try:
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_name}'.")
except Exception as e:
    print(f"An error occurred: {e}")
# ==========================================
# 1. ЗАВАНТАЖЕННЯ МОДЕЛІ ТА СКЕЙЛЕРІВ
# ==========================================
print("Завантаження даних для ініціалізації скейлерів...")
df_train = pd.read_csv('fem_stress_dataset_method_b_2000.csv') # ТУТ ПРАВИЛЬНИЙ ФАЙЛ

y_cols = [col for col in df_train.columns if col.startswith('TARGET_')]
ignore_cols = ['sample_id', 'r', 'z', 'p'] + y_cols
x_cols = [col for col in df_train.columns if col not in ignore_cols]

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
scaler_X.fit(df_train[x_cols].values)
scaler_Y.fit(df_train[y_cols].values)

class StressPredictorNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(StressPredictorNN, self).__init__()
        # Твоя найкраща архітектура 512-512-512!
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2), 
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.1), 
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, output_size)
        )
    def forward(self, x):
        return self.network(x)

model = StressPredictorNN(input_size=len(x_cols), output_size=len(y_cols))
# Переконайся, що завантажуєш правильну натреновану модель!
model.load_state_dict(torch.load("fem_stress_model_method_b.pth"))
model.eval()

# ==========================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ==========================================
def format_stress_dict(stress_array, prefix=""):
    return {f"{prefix}srr": stress_array[0], f"{prefix}szz": stress_array[1],
            f"{prefix}srz": stress_array[2], f"{prefix}stt": stress_array[3]}

def node_id_is_on_boundary(mesh, node_id):
    bnd_nodes = set(mesh.leftBoundaryNodes + mesh.rightBoundaryNodes + mesh.topBoundaryNodes + mesh.bottomBoundaryNodes)
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

# ==========================================
# 2. APPLICATION PHASE (METHOD-B)
# ==========================================
test_p = 135.0
print(f"\nЗапуск нового розрахунку МСЕ (Method-B) з тиском p={test_p}...")

r_min, r_max, z_min, z_max = 1.0, 2.0, 0.0, 1.0
nu, mu_base = 0.3, 0.7
mat = Material("Steel", E=2 * mu_base * (1 + nu), nu=nu)

config = ExperimentConfig(
    r_min=r_min, r_max=r_max, z_min=z_min, z_max=z_max, r_func=lambda z: 0,
    mesh_resolutions=[], p=test_p, nu=nu, mu=mu_base,
    fixed_z=0.5, fixed_r=1, compare_eps=1e-6,
    element_types=[ElementType.LINEAR], n_points_range=[2]
)

work_dir = f"temp_eval_{os.getpid()}"
os.makedirs(work_dir, exist_ok=True)
original_cwd = os.getcwd()
os.chdir(work_dir)

results_for_plot = []

try:
    # --- ЦИКЛ 0: Груба сітка ---
    factory_c = FEMFactory(r_min, r_max, z_min, z_max, 0, 0, material=mat, node_dof=2).init(ElementType.LINEAR)
    _, mesh_c = factory_c.create(n_points=2)
    mesh_c.generate(config.r_min, config.r_max, config.z_min, config.z_max, rN=4, zN=4, r_func=config.r_func)
    
    bcs_c0 = create_boundary_conditions(mesh_c, test_p)
    solver_c0 = AxisymmetricFEMSolver(mesh_c, bcs_c0)
    solver_c0.run(custom_n_points=2, element_type=ElementType.LINEAR)
    
    pp_raw_c0 = FEMPostProcessor(mesh_c, recovery_mode=RECOVERY_RAW)
    
    target_nodes = []
    for target_node_id, node in list(mesh_c.nodes.items()):
        if node_id_is_on_boundary(mesh_c, target_node_id): continue
        neighbors = get_4_neighbors(mesh_c, target_node_id)
        if len(neighbors) != 4: continue
        
        S0_T = pp_raw_c0.stresses_at(np.array([[node.r, node.z]]))[0]
        
        n_info = []
        for n_id in neighbors:
            n_node = mesh_c.nodes[n_id]
            S0_N = pp_raw_c0.stresses_at(np.array([[n_node.r, n_node.z]]))[0]
            n_info.append({'r': n_node.r, 'z': n_node.z, 'S0_N': S0_N})
            
        target_nodes.append({'id': target_node_id, 'r': node.r, 'z': node.z, 'S0_T': S0_T, 'neighbors': n_info})

    # --- ЦИКЛ 1: ОДИН КРОК АДАПТАЦІЇ ---
    ibem_strategy = FindElementsForRefinementIBEM(io=NullIOService())
    refine_ids = ibem_strategy.find(mesh_c, threshold=0.3, bcs=bcs_c0, mode='eta')
    
    if refine_ids:
        mesh_c.refine_elements(refine_ids, auto_plot=False)
        bcs_c1 = create_boundary_conditions(mesh_c, test_p)
        solver_c1 = AxisymmetricFEMSolver(mesh_c, bcs_c1)
        mesh_c.build_and_attach_mortar_interfaces(solver_c1)
        solver_c1.run(custom_n_points=2, element_type=ElementType.LINEAR)
        
    pp_raw_c1 = FEMPostProcessor(mesh_c, recovery_mode=RECOVERY_RAW)

    # --- ЕТАЛОН: Дрібна сітка (32x32) ---
    factory_f = FEMFactory(r_min, r_max, z_min, z_max, 0, 0, material=mat, node_dof=2).init(ElementType.LINEAR)
    _, mesh_f = factory_f.create(n_points=2)
    mesh_f.generate(config.r_min, config.r_max, config.z_min, config.z_max, rN=32, zN=32, r_func=config.r_func)
    bcs_f = create_boundary_conditions(mesh_f, test_p)
    solver_f = AxisymmetricFEMSolver(mesh_f, bcs_f)
    solver_f.run(custom_n_points=2, element_type=ElementType.LINEAR)
    pp_f = FEMPostProcessor(mesh_f, recovery_mode=RECOVERY_SPR)

    # ==========================================
    # 3. ФОРМУВАННЯ X ТА ПРОГНОЗ
    # ==========================================
    print("Застосування Нейромережі (Method-B)...")
    
    for info in target_nodes:
        pt = np.array([[info['r'], info['z']]])
        S1_T = pp_raw_c1.stresses_at(pt)[0]
        F_T = pp_f.stresses_at(pt)[0]
        
        row_data = {}
        row_data.update(format_stress_dict(info['S0_T'] - S1_T, prefix="PT_diff_"))
        
        for idx, n_info in enumerate(info['neighbors']):
            n_pt = np.array([[n_info['r'], n_info['z']]])
            S1_N = pp_raw_c1.stresses_at(n_pt)[0]
            
            row_data.update(format_stress_dict(n_info['S0_N'] - S1_N, prefix=f"PN{idx+1}_diff_"))
            row_data.update(format_stress_dict(n_info['S0_N'] - info['S0_T'], prefix=f"PN{idx+1}_var_"))

        df_infer = pd.DataFrame([row_data])
        X_infer = df_infer[x_cols].values
        X_scaled = scaler_X.transform(X_infer)
        
        with torch.no_grad():
            pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32))
            pred_real = scaler_Y.inverse_transform(pred_scaled.numpy())[0]
            
        NN_T = info['S0_T'] + pred_real
        
        results_for_plot.append({
            'r': info['r'], 'z': info['z'],
            'srr_coarse': info['S0_T'][0], 'srr_fine': F_T[0], 'srr_nn': NN_T[0],
            'szz_coarse': info['S0_T'][1], 'szz_fine': F_T[1], 'szz_nn': NN_T[1],
            'srz_coarse': info['S0_T'][2], 'srz_fine': F_T[2], 'srz_nn': NN_T[2],
            'stt_coarse': info['S0_T'][3], 'stt_fine': F_T[3], 'stt_nn': NN_T[3]
        })

finally:
    os.chdir(original_cwd)
    shutil.rmtree(work_dir, ignore_errors=True)

# ==========================================
# 4. ВІЗУАЛІЗАЦІЯ В 8 ОКРЕМИХ ФАЙЛІВ
# ==========================================
print("Будуємо графіки...")
res_df = pd.DataFrame(results_for_plot)
target_z = 0.75
slice_df = res_df[abs(res_df['z'] - target_z) < 1e-3].sort_values(by='r')

components = [
    ('srr', r'\sigma_{rr}', 'sigma_rr'),
    ('szz', r'\sigma_{zz}', 'sigma_zz'),
    ('stt', r'\sigma_{\theta\theta}', 'sigma_tt'),
    ('srz', r'\sigma_{rz}', 'sigma_rz')
]

for comp, tex, filename in components:
    # 1. Графік абсолютних значень
    plt.figure(figsize=(8, 6))
    plt.plot(slice_df['r'], slice_df[f'{comp}_coarse'], 'ro--', markersize=8, label=f'${tex}$ (Coarse Mesh)')
    plt.plot(slice_df['r'], slice_df[f'{comp}_fine'], 'g^:', markersize=8, label=f'${tex}$ (Fine Mesh - Target)')
    plt.plot(slice_df['r'], slice_df[f'{comp}_nn'], 'b-', linewidth=3, label=f'${tex}$ (Deep Learning)')
    plt.title(f'Absolute Stress ${tex}$', fontsize=14)
    plt.xlabel('Radius (r)', fontsize=12)
    plt.ylabel('Stress (Pa)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{directory_name}/{filename}.png', dpi=300)
    plt.close()

    # 2. Графік похибки
    err_coarse = slice_df[f'{comp}_coarse'] - slice_df[f'{comp}_fine']
    err_nn = slice_df[f'{comp}_nn'] - slice_df[f'{comp}_fine']
    
    plt.figure(figsize=(8, 6))
    plt.plot(slice_df['r'], err_coarse, 'ro--', markersize=8, label='Error (Coarse Mesh)')
    plt.plot(slice_df['r'], err_nn, 'b-', linewidth=3, label='Error (Deep Learning)')
    plt.axhline(0, color='green', linestyle=':', linewidth=2)
    plt.title(fr'Error Distribution $\Delta{tex}$', fontsize=14)
    plt.xlabel('Radius (r)', fontsize=12)
    plt.ylabel('Error (Pa)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{directory_name}/error_{filename}.png', dpi=300)
    plt.close()

print("Успішно збережено 8 графіків: sigma_rr.png, error_sigma_rr.png, ... і т.д.")

import numpy as np
import matplotlib.pyplot as plt
from wavecore import simulate_wave, save_animation, compute_errors

def analyze_point(frames, dt, x, z, N, threshold=1e-7):
    i = int(z * N)
    j = int(x * N)
    values = [frame[i, j] for frame in frames]
    max_amp = np.max(np.abs(values))
    for idx, v in enumerate(values):
        if abs(v) > threshold:
            return idx * dt, max_amp
    return None, max_amp

def estimate_wavefront_angle(frame, N, threshold=1e-2):
    maxval = np.max(np.abs(frame))
    mask = np.abs(frame) > threshold * maxval
    indices = np.argwhere(mask)
    if len(indices) < 2:
        return None
    x = indices[:,1] / N
    z = indices[:,0] / N
    p = np.polyfit(x, z, 1)
    angle_rad = np.arctan(p[0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def multiple_interfaces_example():
    N = 256
    T = 1.0
    f0 = 14.0

    c_field = cfield_multiple_interfaces(N)
    src_x = 0.25
    src_z = 0.5
    src_i = int(src_z * N)
    src_j = int(src_x * N)

    u, frames, dt = simulate_wave(
        N, T, f0=f0, boundary='fixed', collect_frames=True, c_field=c_field,
        src_i=src_i, src_j=src_j
    )

    fps = 5
    steps_per_frame = max(1, int(0.5 / (fps * dt)))
    save_animation(frames, dt, steps_per_frame, "multiple_interfaces_hard.gif", cmap='viridis')
    points = [(0.16, 0.5), (0.5, 0.5), (0.83, 0.5)]
    print("\nАналіз точок (множинні інтерфейси, ускладнений):")
    for x, z in points:
        t_hit, max_amp = analyze_point(frames, dt, x, z, N)
        print(f"Точка ({x:.2f}, {z:.2f}): час досягнення = {t_hit}, макс. амплітуда = {max_amp:.3e}")
    angle = estimate_wavefront_angle(frames[-1], N)
    print(f"Оцінка кута фронту хвилі (ост. кадр): {angle:.2f} градусів")

def inclined_interface_example():
    N = 256
    T = 1.0
    f0 = 18.0

    x_vals = np.linspace(0, 1, N+1)
    z_vals = np.linspace(0, 1, N+1)
    X, Z = np.meshgrid(x_vals, z_vals)
    interface = 0.5 + 0.3 * (X - 0.5)
    c_field = np.ones((N+1, N+1))
    c_field[Z < interface] = 0.9
    c_field[Z >= interface] = 2.6

    src_x = 0.3
    src_z = 0.3
    src_i = int(src_z * N)
    src_j = int(src_x * N)

    u, frames, dt = simulate_wave(
        N, T, f0=f0, boundary='fixed', collect_frames=True, c_field=c_field,
        src_i=src_i, src_j=src_j
    )

    fps = 5
    steps_per_frame = max(1, int(0.5 / (fps * dt)))
    save_animation(frames, dt, steps_per_frame, "inclined_interface_hard.gif", cmap='viridis')

    points = [(0.2, 0.2), (0.8, 0.8), (0.5, 0.5)]
    print("\nАналіз точок (ускладнений похилий інтерфейс):")
    for x, z in points:
        t_hit, max_amp = analyze_point(frames, dt, x, z, N)
        print(f"Точка ({x:.2f}, {z:.2f}): час досягнення = {t_hit}, макс. амплітуда = {max_amp:.3e}")
    angle = estimate_wavefront_angle(frames[-1], N)
    print(f"Оцінка кута фронту хвилі (ост. кадр): {angle:.2f} градусів")

def cfield_multiple_interfaces(N):
    x_vals = np.linspace(0, 1, N+1)
    c_field = np.ones((N+1, N+1))
    c_field[:, x_vals < 0.33] = 0.8
    c_field[:, (x_vals >= 0.33) & (x_vals < 0.66)] = 2.5
    c_field[:, x_vals >= 0.66] = 1.2
    return c_field

def cfield_inclined_interface(N):
    x_vals = np.linspace(0, 1, N+1)
    z_vals = np.linspace(0, 1, N+1)
    X, Z = np.meshgrid(x_vals, z_vals)
    interface = 0.5 + 0.3 * (X - 0.5)
    c_field = np.ones((N+1, N+1))
    c_field[Z < interface] = 1.0
    c_field[Z >= interface] = 2.0
    return c_field

def cfield_classic(N):
    x_vals = np.linspace(0, 1, N+1)
    c1 = 1.0
    c2 = 2.0
    c_field = np.where(x_vals < 0.5, c1, c2)
    c_field = np.tile(c_field, (N+1, 1))
    return c_field

def classic_example():
    N = 256
    T = 1.0
    f0 = 4.0
    c_field = cfield_classic(N)
    u, frames, dt = simulate_wave(N, T, f0=f0, boundary='fixed', collect_frames=True, c_field=c_field)
    fps = 5
    steps_per_frame = max(1, int(0.5 / (fps * dt)))
    save_animation(frames, dt, steps_per_frame, "classic_example.gif", cmap='viridis')
    points = [(0.25, 0.5), (0.5, 0.5), (0.75, 0.5)]
    print("\nАналіз точок (класичний приклад):")
    for x, z in points:
        t_hit, max_amp = analyze_point(frames, dt, x, z, N)
        print(f"Точка ({x:.2f}, {z:.2f}): час досягнення = {t_hit}, макс. амплітуда = {max_amp:.3e}")
    angle = estimate_wavefront_angle(frames[-1], N)
    print(f"Оцінка кута фронту хвилі (ост. кадр): {angle:.2f} градусів")

def convergence_study(example_name, cfield_func):
    print(f"\n=== Збіжність для прикладу: {example_name} ===")
    grid_sizes = [64, 128, 256, 512]
    T = 1.0
    f0 = 4.0
    boundary = 'fixed'
    N_ref = 1024
    print(f"Обчислення референсного розв'язку на {N_ref}x{N_ref}...")
    c_field_ref = cfield_func(N_ref)
    u_ref, _, _ = simulate_wave(N_ref, T, f0=f0, boundary=boundary, collect_frames=False, c_field=c_field_ref)
    dx_ref = 1.0 / N_ref
    errors = {}
    dxs = {}
    for N in grid_sizes:
        print(f"Сітка {N}x{N}...")
        c_field = cfield_func(N)
        u, frames, dt = simulate_wave(N, T, f0=f0, boundary=boundary, collect_frames=True, c_field=c_field)
        dx = 1.0 / N
        L2_err, Linf_err, MAE_err = compute_errors(u_ref, dx_ref, u, dx)
        errors[N] = (L2_err, Linf_err, MAE_err)
        dxs[N] = dx
        fps = 5
        steps_per_frame = max(1, int(0.5 / (fps * dt)))
        save_animation(frames, dt, steps_per_frame, f"{example_name}_{N}x{N}.gif", cmap='viridis')
    table_lines = []
    header = f"{'Grid':>8} | {'dx':>8} | {'L2 Error':>10} | {'L2 order':>8} | {'Linf':>10} | {'Linf order':>8} | {'MAE':>10} | {'MAE order':>8}"
    table_lines.append(header)
    table_lines.append("-" * len(header))
    prev_L2 = prev_Linf = prev_MAE = prev_dx = None
    for N in grid_sizes:
        L2_err, Linf_err, MAE_err = errors[N]
        dx = dxs[N]
        if prev_L2 is not None:
            L2_order = np.log(prev_L2 / L2_err) / np.log(prev_dx / dx)
            Linf_order = np.log(prev_Linf / Linf_err) / np.log(prev_dx / dx)
            MAE_order = np.log(prev_MAE / MAE_err) / np.log(prev_dx / dx)
        else:
            L2_order = Linf_order = MAE_order = float('nan')
        line = f"{N:>6}x{N:<2} | {dx:8.5f} | {L2_err:10.3e} | {L2_order:8.2f} | {Linf_err:10.3e} | {Linf_order:8.2f} | {MAE_err:10.3e} | {MAE_order:8.2f}"
        table_lines.append(line)
        prev_L2, prev_Linf, prev_MAE, prev_dx = L2_err, Linf_err, MAE_err, dx
    table = "\n".join(table_lines)
    print("\nТаблиця збіжності:")
    print(table)
    with open("convergence_table.txt", "w") as f:
        f.write(table)

def convergence_study_classic():
    convergence_study("classic_example", cfield_classic)

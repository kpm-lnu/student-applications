import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def ricker_wavelet(f0, t):
    a = np.pi * f0 * t
    return (1.0 - 2.0 * a ** 2) * np.exp(-a ** 2)

def simulate_wave(N, T, c=1.0, f0=4.0, boundary='fixed', collect_frames=False, c_field=None, src_i=None, src_j=None):
    Npoints = N + 1
    dx = 1.0 / N
    dz = dx
    if c_field is None:
        c1 = 1.0
        c2 = 2.0
        c_max = max(c1, c2)
        x_vals = np.linspace(0, 1, Npoints)
        c_field = np.where(x_vals < 0.5, c1, c2)
        c_field = np.tile(c_field, (Npoints, 1))
    else:
        c_max = np.max(c_field)
    dt = dx / (np.sqrt(2) * c_max)
    num_steps = int(np.ceil(T / dt))
    T_final = num_steps * dt
    u_prev = np.zeros((Npoints, Npoints))
    u_cur = np.zeros((Npoints, Npoints))
    if src_i is None or src_j is None:
        src_i = src_j = Npoints // 2
    frames = []
    if collect_frames:
        fps = 10
        steps_per_frame = max(1, int(1.0 / (fps * dt)))
    else:
        steps_per_frame = None
    t = 0.0
    for n in range(num_steps):
        force = ricker_wavelet(f0, t)
        u_next = np.zeros_like(u_cur)
        c2 = c_field ** 2
        ux_plus = u_cur[1:-1, 2:] - u_cur[1:-1, 1:-1]
        ux_minus = u_cur[1:-1, 1:-1] - u_cur[1:-1, 0:-2]
        c2_x_plus = 0.5 * (c2[1:-1, 2:] + c2[1:-1, 1:-1])
        c2_x_minus = 0.5 * (c2[1:-1, 1:-1] + c2[1:-1, 0:-2])
        lap_x = (c2_x_plus * ux_plus - c2_x_minus * ux_minus) / (dx ** 2)
        uz_plus = u_cur[2:, 1:-1] - u_cur[1:-1, 1:-1]
        uz_minus = u_cur[1:-1, 1:-1] - u_cur[0:-2, 1:-1]
        c2_z_plus = 0.5 * (c2[2:, 1:-1] + c2[1:-1, 1:-1])
        c2_z_minus = 0.5 * (c2[1:-1, 1:-1] + c2[0:-2, 1:-1])
        lap_z = (c2_z_plus * uz_plus - c2_z_minus * uz_minus) / (dz ** 2)
        laplacian = lap_x + lap_z
        u_next[1:-1, 1:-1] = (2.0 * u_cur[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + (dt ** 2) * laplacian)
        if boundary == 'fixed':
            u_next[0, :] = 0.0
            u_next[-1, :] = 0.0
            u_next[:, 0] = 0.0
            u_next[:, -1] = 0.0
        elif boundary == 'free':
            u_next[0, :] = u_next[1, :]
            u_next[-1, :] = u_next[-2, :]
            u_next[:, 0] = u_next[:, 1]
            u_next[:, -1] = u_next[:, -2]
        u_next[src_i, src_j] += force * (dt ** 2)
        if collect_frames and (n % steps_per_frame == 0 or n == num_steps - 1):
            frames.append(u_cur.copy())
        t += dt
        u_prev, u_cur = u_cur, u_next
    return u_cur, frames, dt

def save_animation(frames, dt, steps_per_frame, filename, cmap='viridis'):
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, origin='lower', extent=[0, 1, 0, 1])
    ax.set_title("Розповсюдження хвиль (u_z)")
    plt.colorbar(im, ax=ax)
    def update(frame):
        im.set_array(frame)
        return [im]
    ani = animation.ArtistAnimation(
        fig,
        [[ax.imshow(frame, cmap=cmap, origin='lower', extent=[0, 1, 0, 1])]
         for frame in frames],
        interval=dt * steps_per_frame * 1000,
        blit=True
    )
    writer = animation.PillowWriter(fps=10)
    ani.save(filename, writer=writer)
    plt.close(fig)
    print(f"Animation saved as {filename}")

def compute_errors(u_ref, dx_ref, u_coarse, dx_coarse):
    ratio = int(round(dx_coarse / dx_ref))
    u_ref_coarse = u_ref[::ratio, ::ratio]
    error = u_coarse - u_ref_coarse
    L2 = np.sqrt(np.sum(error ** 2) * (dx_coarse ** 2))
    Linf = np.max(np.abs(error))
    MAE = np.mean(np.abs(error))
    return L2, Linf, MAE 
"""
Advection-diffusion-reaction pde:

  du/dt - D(dot(grad(u), grad(u))) + dot(v, grad(u)) + Ku = f

"""

import tkinter as tk
from tkinter import messagebox
from fenics import *

from advection_diffusion_reaction_pde import solve_adr_pde
from adr_types import AdrParams, Velocity, Rectangle

def run_simulation():
    try:
        D = Constant(float(entry_D.get()))
        K = Constant(float(entry_K.get()))
        f_str = str(entry_f.get())
        u_exact_str = str(entry_u_exact.get())

        adr_test_params1 = AdrParams(
            diffusion=D,
            reaction=K,
            velocity=Velocity(
                float(entry_vx.get()),
                float(entry_vy.get())
            ),
            f_source=Expression(f_str, degree=4, t=0, D=D, K=K, label=f_str),
            plane=Rectangle(float(entry_a.get()), float(entry_b.get())),
            N_mesh=int(entry_N.get()),
            T_time=float(entry_T.get()),
            M_time_iter=int(entry_M.get()),
            u_exact=Expression(u_exact_str, degree=4, t=0, D=D, K=K, label=u_exact_str)
        )

        solve_adr_pde(adr_test_params1)
        root.destroy()

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.geometry("800x800")

root.title("ЧИСЛОВЕ РОЗВ’ЯЗУВАННЯ ДВОВИМІРНИХ ЗАДАЧ АДВЕКЦІЇ–ДИФУЗІЇ–РЕАКЦІЇ")

# Example 1
# du/dt - 0.001(dot(grad(u), grad(u))) + dot([1,1], grad(u)) + u = 0
# u_exact = e^((1/2)*(0.001*t - x - y))
fields = [
    ("Diffusion D", "0.001"),
    ("Reaction K", "1.0"),
    ("Velocity v_x", "1.0"),
    ("Velocity v_y", "1.0"),
    ("Source function f", "0"),
    ("Grid size N", "40"),
    ("Final time T", "10.0"),
    ("Time steps M", "40"),
    ("Domain size a", "1.0"),
    ("Domain size b", "1.0"),
    ("Exact solution", "exp((D*t -x[0] - x[1])/2)")
]

# Example 2:
# du/dt - 0.001(dot(grad(u), grad(u))) + dot([1,2], grad(u)) + 2u = (1/2)*e^((1/2)*(0.001*t - x - y))
# u_exact = e^((1/2)*(0.001*t - x - y))
'''
fields = [
    ("Diffusion D", "0.1"),
    ("Reaction K", "2"),
    ("Velocity v_x", "1.0"),
    ("Velocity v_y", "2.0"),
    ("Source function f", "exp((D*t -x[0] - x[1])/2)/2"),
    ("Grid size N", "80"),
    ("Final time T", "10.0"),
    ("Time steps M", "40"),
    ("Domain size a", "1.0"),
    ("Domain size b", "1.0"),
    ("Exact solution", "exp((D*t -x[0] - x[1])/2)")
]
'''

# Example 3:
# du/dt - 0.001(dot(grad(u), grad(u))) + dot([1,2], grad(u)) + 2u = (1/2)*e^((1/2)*(0.001*t - x - y))
# u_exact = e^((1/2)*(0.001*t - x - y))
'''
fields = [
    ("Diffusion D", "0.1"),
    ("Reaction K", "1"),
    ("Velocity v_x", "1.0"),
    ("Velocity v_y", "0.0"),
    ("Source function f", "'exp((D*t -x[0] - x[1])/2)/2"),
    ("Grid size N", "80"),
    ("Final time T", "10.0"),
    ("Time steps M", "40"),
    ("Domain size a", "1.0"),
    ("Domain size b", "1.0"),
    ("Exact solution", "exp((D*t -x[0] - x[1])/2)")
]
'''

entries = []
for i, (label, default) in enumerate(fields):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(root)
    entry.insert(0, default)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

entry_D, entry_K, entry_vx, entry_vy, entry_f, entry_N, entry_T, entry_M, entry_a, entry_b, entry_u_exact = entries

# Run button
tk.Button(root, text="Run Simulation", command=run_simulation, bg="green", fg="white").grid(row=len(fields), columnspan=2, pady=10)

root.mainloop()


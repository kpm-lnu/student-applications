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
        
        adr_test_params1 = AdrParams(
            diffusion=D,
            reaction=K,
            velocity=Velocity(
                Constant(float(entry_vx.get())),
                Constant(float(entry_vy.get()))
            ),
            f_source=Expression(str(entry_f.get()), degree=4, t=0, D=D, K=K, label=str(entry_f.get())),
            plane=Rectangle(float(entry_a.get()), float(entry_b.get())),
            N_mesh=int(entry_N.get()),
            T_time=float(entry_T.get()),
            M_time_iter=int(entry_M.get()),
            boundary=[
                Expression(str(entry_boundary_x1.get()), degree=4, t=0, D=D, K=K, label=str(entry_boundary_x1.get())),
                Expression(str(entry_boundary_x1.get()), degree=4, t=0, D=D, K=K, label=str(entry_boundary_x1.get())),
                Expression(str(entry_boundary_y0.get()), degree=4, t=0, D=D, K=K, label=str(entry_boundary_y0.get())),
                Expression(str(entry_boundary_y1.get()), degree=4, t=0, D=D, K=K, label=str(entry_boundary_y1.get()))
            ],
            initial=Expression(str(entry_init.get()), degree=4, t=0, D=D, K=K, label=str(entry_init.get())),
            u_exact=Expression( str(entry_u_exact.get()), degree=4, t=0, D=D, K=K, label= str(entry_u_exact.get()))
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

# fields = [
#     ("Diffusion D", "0.001"),
#     ("Reaction K", "1.0"),
#     ("Velocity v_x", "1.0"),
#     ("Velocity v_y", "1.0"),
#     ("Source function f", "0"),
#     ("Grid size N", "40"),
#     ("Final time T", "10.0"),
#     ("Time steps M", "40"),
#     ("Domain size a", "1.0"),
#     ("Domain size b", "1.0"),
#     ("Boundary condition x0", "exp((D*t -x[0] - x[1])/2)"),
#     ("Boundary condition x1", "exp((D*t -x[0] - x[1])/2)"),
#     ("Boundary condition y0", "exp((D*t -x[0] - x[1])/2)"),
#     ("Boundary condition y1", "exp((D*t -x[0] - x[1])/2)"),
#     ("Initial condition", "exp((D*t -x[0] - x[1])/2)"),
#     ("Exact solution", "exp((D*t -x[0] - x[1])/2)")
# ]

# Example 2:
# du/dt - 0.1(dot(grad(u), grad(u))) + dot([1,2], grad(u)) + 2u = (1/2)*e^((1/2)*(0.1*t - x - y))
# u_exact = e^((1/2)*(0.1*t - x - y))

fields = [
    ("Diffusion D", "0.1"),
    ("Reaction K", "2"),
    ("Velocity v_x", "1.0"),
    ("Velocity v_y", "2.0"),
    ("Source function f", "exp((D*t -x[0] - x[1])/2)/2"),
    ("Grid size N", "40"),
    ("Final time T", "10.0"),
    ("Time steps M", "40"),
    ("Domain size a", "1.0"),
    ("Domain size b", "1.0"),
    ("Boundary condition x0", "exp((D*t -x[0] - x[1])/2)"),
    ("Boundary condition x1", "exp((D*t -x[0] - x[1])/2)"),
    ("Boundary condition y0", "exp((D*t -x[0] - x[1])/2)"),
    ("Boundary condition y1", "exp((D*t -x[0] - x[1])/2)"),
    ("Initial condition", "exp((D*t -x[0] - x[1])/2)"),
    ("Exact solution", "exp((D*t -x[0] - x[1])/2)")
]

# Example 3:
# du/dt - 0.1(dot(grad(u), grad(u))) + dot([1,0], grad(u)) + u = (1/2)*e^((1/2)*(0.1*t - x - y))
# u_exact = e^((1/2)*(0.1*t - x - y))

'''
fields = [
    ("Diffusion D", "0.1"),
    ("Reaction K", "1"),
    ("Velocity v_x", "1.0"),
    ("Velocity v_y", "0.0"),
    ("Source function f", "'exp((D*t -x[0] - x[1])/2)/2"),
    ("Grid size N", "40"),
    ("Final time T", "10.0"),
    ("Time steps M", "40"),
    ("Domain size a", "1.0"),
    ("Domain size b", "1.0"),
    ("Boundary condition x0", "exp((D*t -x[0] - x[1])/2)"),
    ("Boundary condition x1", "exp((D*t -x[0] - x[1])/2)"),
    ("Boundary condition y0", "exp((D*t -x[0] - x[1])/2)"),
    ("Boundary condition y1", "exp((D*t -x[0] - x[1])/2)"),
    ("Initial condition", "exp((D*t -x[0] - x[1])/2)"),
    ("Exact solution", "exp((D*t -x[0] - x[1])/2)")
]
'''

entries = []
for i, (label, default) in enumerate(fields):
    tk.Label(root, text=label).grid(row=i, column=0, padx=20, pady=10, sticky="w")
    entry = tk.Entry(root, width=30)
    entry.insert(0, default)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

entry_D, entry_K, \
    entry_vx, entry_vy, entry_f, \
    entry_N, entry_T, entry_M, \
    entry_a, entry_b, \
    entry_boundary_x0, entry_boundary_x1, entry_boundary_y0, entry_boundary_y1, \
    entry_init, entry_u_exact = entries

tk.Button(root, text="Розв'язати", command=run_simulation, bg="green", fg="white").grid(row=len(fields), columnspan=2, pady=10)

root.mainloop()

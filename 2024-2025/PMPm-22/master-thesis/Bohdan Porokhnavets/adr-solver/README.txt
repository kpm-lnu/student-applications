Advection-diffusion-reaction PDE solver

PDE: du/dt - D(dot(grad(u), grad(u))) + dot(v, grad(u)) + Ku = f



dolfin - ADR solver based on legacy FEniCS, more stable
    To run GUI: "python3 dolfin/advection_diffusion_reaction_pde_gui.py"
    To run main test: "python3 dolfin/advection_diffusion_reaction_pde.py"


dolfinx - ADR solver based on next-gen FEniCSx
    To run GUI: "python3 dolfinx/advection_diffusion_reaction_pde_gui.py"
    To run main test: "python3 dolfinx/advection_diffusion_reaction_pde.py"

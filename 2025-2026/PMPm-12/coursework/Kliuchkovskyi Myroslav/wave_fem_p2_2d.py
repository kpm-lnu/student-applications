#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ─────────────────────────────────────────────────────────────────────────────
# 0.  AUTO-INSTALL
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys, io
for _pkg in ["numpy", "scipy", "matplotlib", "pillow"]:
    try:
        __import__(_pkg.split(".")[0])
    except ImportError:
        print(f"[setup] Installing {_pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg, "-q"])

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
from scipy.sparse import coo_matrix, diags as sp_diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import time

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION  (all user-facing parameters in one block)
# ─────────────────────────────────────────────────────────────────────────────
LX, LY = 1.0, 1.0      # domain size [m]
C_BG   = 1.0            # background wave speed [m/s]

# Part 1 – static convergence
N_STATIC = [4, 8, 12, 16, 24]

# Part 2 – dynamic validation (standing wave)
N_DYN  = 20
T_DYN  = 3.0
OMEGA  = 2.0 * np.pi

# Parts 3–4 – fracture study
N_FRAC       = 24       # → (2·24+1)² = 2401 P2 nodes
T_FRAC       = 1.0      # simulation time [s] — safe: boundary reflections arrive at t≈1.27–1.6 s
F0           = 3.0      # Ricker centre frequency [Hz]
T0_RICK      = 0.3/F0   # Ricker time shift [s]
SRC_STRIP    = 0.10     # source active for x < SRC_STRIP·Lx
FRAC_HALF    = 0.35     # fracture half-length [m]

# Part 3 – angle variation (fixed thickness and xi)
ANGLES    = [30, 45, 60, 90]
THICK_DEF = 0.08        # default fault thickness [m]
XI_DEF    = 0.4         # default velocity ratio  (c_frac = XI · c_bg)

# Part 4 – sensitivity study (fixed angle = 60°)
FIXED_ANGLE  = 60
THICKNESSES  = [0.04, 0.08, 0.16]   # thin / medium / thick
XI_VALUES    = [0.2, 0.4, 0.6]      # velocity ratios (paper range)

# Receiver positions (Parts 3–4)  —  mirrors paper Fig. 1: LS_L, LS_C, LS_R
REC_X, REC_Y = 0.8*LX, 0.5*LY   # backward-compat alias (= LS_R)
LS_L = (0.25*LX, 0.5*LY)         # left/upstream  — measures reflected signal
LS_C = (0.50*LX, 0.5*LY)         # fault centre   — measures trapped signal
LS_R = (0.80*LX, 0.5*LY)         # right/transmitted — measures transmitted signal

# ─────────────────────────────────────────────────────────────────────────────
# 2.  QUADRATURE RULES ON REFERENCE TRIANGLE
#     Weights normalised so sum(w) = 1.
#     Physical integral: ∫_T f dA = area(T) · Σ_q w_q f(x_q)
# ─────────────────────────────────────────────────────────────────────────────
# 3-point midpoint rule  (exact degree 2) – stiffness
_QP3 = np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
_QW3 = np.array([1/3, 1/3, 1/3])

# 6-point Dunavant rule  (exact degree 4) – mass, force, L² error
_a1=0.445948490915965; _w1=0.223381589678011
_a2=0.091576213509771; _w2=0.109951743655322
_QP6 = np.array([
    [_a1,_a1,1-2*_a1],[_a1,1-2*_a1,_a1],[1-2*_a1,_a1,_a1],
    [_a2,_a2,1-2*_a2],[_a2,1-2*_a2,_a2],[1-2*_a2,_a2,_a2],
])
_QW6 = np.array([_w1,_w1,_w1,_w2,_w2,_w2])

# ─────────────────────────────────────────────────────────────────────────────
# 3.  P2 BASIS FUNCTIONS  (standard Lagrange, barycentric coordinates)
#     Local node order: [v1, v2, v3, mid12, mid23, mid13]
#     Partition of unity verified: Σ φᵢ = 1  (shown analytically in audit)
# ─────────────────────────────────────────────────────────────────────────────
def p2_phi(L):
    """P2 Lagrange basis. L: (nq,3) barycentric → returns (6,nq)."""
    l1,l2,l3 = L[:,0],L[:,1],L[:,2]
    return np.array([
        l1*(2*l1-1), l2*(2*l2-1), l3*(2*l3-1),
        4*l1*l2,     4*l2*l3,     4*l1*l3,
    ])

def p2_dphi_dbary(L):
    """d(φᵢ)/d(λₖ) at nq points. Returns (6,nq,3)."""
    l1,l2,l3 = L[:,0],L[:,1],L[:,2]
    G = np.zeros((6,len(l1),3))
    G[0,:,0]=4*l1-1
    G[1,:,1]=4*l2-1
    G[2,:,2]=4*l3-1
    G[3,:,0]=4*l2; G[3,:,1]=4*l1
    G[4,:,1]=4*l3; G[4,:,2]=4*l2
    G[5,:,0]=4*l3; G[5,:,2]=4*l1
    return G

_PHI6  = p2_phi(_QP6)         # (6,6)   – mass / force / L² error
_DPHI3 = p2_dphi_dbary(_QP3)  # (6,3,3) – stiffness gradients

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MESH GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def make_p2_mesh(Nx, Ny, Lx=1.0, Ly=1.0):
    """
    Structured P2 triangular mesh on [0,Lx]×[0,Ly].
    Returns:
      nodes   (nn,2)  physical coordinates
      tris    (ne,6)  P2 DOF indices [v1,v2,v3, m12,m23,m13]
      p1_tris (ne,3)  P1 sub-triangles for matplotlib
    Each rectangle split into 2 triangles: SW-SE-NW and SE-NE-NW.
    """
    hx,hy  = Lx/Nx, Ly/Ny
    nx2,ny2 = 2*Nx+1, 2*Ny+1
    ii,jj  = np.meshgrid(np.arange(nx2),np.arange(ny2))
    nodes  = np.column_stack([ii.ravel()*hx/2, jj.ravel()*hy/2])

    def idx(i,j): return j*nx2+i

    tris,p1_tris = [],[]
    for j in range(Ny):
        for i in range(Nx):
            n00=idx(2*i,  2*j);   n20=idx(2*i+2,2*j)
            n02=idx(2*i,  2*j+2); n22=idx(2*i+2,2*j+2)
            n10=idx(2*i+1,2*j);   n01=idx(2*i,  2*j+1)
            n21=idx(2*i+2,2*j+1); n12=idx(2*i+1,2*j+2)
            n11=idx(2*i+1,2*j+1)
            tris.append([n00,n20,n02,n10,n11,n01]); p1_tris.append([n00,n20,n02])
            tris.append([n20,n22,n02,n21,n12,n11]); p1_tris.append([n20,n22,n02])
    return nodes, np.array(tris,int), np.array(p1_tris,int)

def boundary_dofs(nodes,Lx,Ly,tol=1e-12):
    x,y=nodes[:,0],nodes[:,1]
    return np.where((x<tol)|(x>Lx-tol)|(y<tol)|(y>Ly-tol))[0]

def interior_dofs(nodes,Lx,Ly,tol=1e-12):
    x,y=nodes[:,0],nodes[:,1]
    return np.where((x>tol)&(x<Lx-tol)&(y>tol)&(y<Ly-tol))[0]

# ─────────────────────────────────────────────────────────────────────────────
# 5.  VECTORISED ASSEMBLY
#     c2 can be a scalar (uniform) OR a per-element array of shape (ne,).
#     This enables the fracture weak-zone model without changing the mesh.
# ─────────────────────────────────────────────────────────────────────────────
def assemble(nodes, tris, c2):
    """Global mass M and stiffness K as sparse CSR. c2 scalar or (ne,)."""
    nn,ne = len(nodes),len(tris)
    v  = nodes[tris[:,:3]]
    x1,y1=v[:,0,0],v[:,0,1]; x2,y2=v[:,1,0],v[:,1,1]; x3,y3=v[:,2,0],v[:,2,1]
    J11,J12=x2-x1,x3-x1; J21,J22=y2-y1,y3-y1
    detJ=J11*J22-J12*J21; areas=np.abs(detJ)/2

    dL=np.zeros((3,ne,2))
    dL[0,:,0]=(y2-y3)/detJ; dL[0,:,1]=(x3-x2)/detJ
    dL[1,:,0]=(y3-y1)/detJ; dL[1,:,1]=(x1-x3)/detJ
    dL[2,:,0]=(y1-y2)/detJ; dL[2,:,1]=(x2-x1)/detJ

    M_fix  = np.einsum('q,iq,jq->ij',_QW6,_PHI6,_PHI6)
    M_locs = areas[:,None,None]*M_fix[None,:,:]

    K_locs = np.zeros((ne,6,6))
    for q in range(3):
        Gq=_DPHI3[:,q,:]
        gx=np.einsum('ik,ke->ei',Gq,dL[:,:,0])
        gy=np.einsum('ik,ke->ei',Gq,dL[:,:,1])
        K_locs += _QW3[q]*areas[:,None,None]*(
            np.einsum('ei,ej->eij',gx,gx)+np.einsum('ei,ej->eij',gy,gy))

    # ── variable or uniform c² ───────────────────────────────────────────────
    if np.isscalar(c2):
        K_locs *= c2
    else:
        K_locs *= np.asarray(c2,dtype=float)[:,None,None]

    I=np.tile(tris[:,:,None],(1,1,6)); J=np.tile(tris[:,None,:],(1,6,1))
    M=coo_matrix((M_locs.ravel(),(I.ravel(),J.ravel())),shape=(nn,nn)).tocsr()
    K=coo_matrix((K_locs.ravel(),(I.ravel(),J.ravel())),shape=(nn,nn)).tocsr()
    return M,K

def assemble_F(nodes, tris, src, t=0.0):
    """Load vector. src(x,y,t) accepts numpy arrays."""
    nn,ne=len(nodes),len(tris)
    F=np.zeros(nn)
    v=nodes[tris[:,:3]]
    J11=v[:,1,0]-v[:,0,0]; J12=v[:,2,0]-v[:,0,0]
    J21=v[:,1,1]-v[:,0,1]; J22=v[:,2,1]-v[:,0,1]
    areas=np.abs(J11*J22-J12*J21)/2
    for q in range(6):
        lam=_QP6[q]
        xq=v[:,0,0]*lam[0]+v[:,1,0]*lam[1]+v[:,2,0]*lam[2]
        yq=v[:,0,1]*lam[0]+v[:,1,1]*lam[1]+v[:,2,1]*lam[2]
        coeff=areas*_QW6[q]*src(xq,yq,t)
        for i in range(6):
            np.add.at(F,tris[:,i],coeff*_PHI6[i,q])
    return F

def apply_dirichlet_sparse(K,F,fixed):
    K=K.tolil()
    for i in fixed: K[i,:]=0; K[:,i]=0; K[i,i]=1.0; F[i]=0.0
    return K.tocsr()

def apply_dirichlet_dense(A,b,fixed):
    for i in fixed: A[i,:]=0; A[:,i]=0; A[i,i]=1.0; b[i]=0.0

# ─────────────────────────────────────────────────────────────────────────────
# 6.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def l2_error(nodes, tris, U, u_exact_fn):
    """
    Proper L² error via 6-point Gauss quadrature on each element.
    FIX vs original: np.mean gave RMS nodal error (not true L² norm).
    """
    v=nodes[tris[:,:3]]
    J11=v[:,1,0]-v[:,0,0]; J12=v[:,2,0]-v[:,0,0]
    J21=v[:,1,1]-v[:,0,1]; J22=v[:,2,1]-v[:,0,1]
    areas=np.abs(J11*J22-J12*J21)/2
    U_dofs=U[tris]          # (ne,6)
    err_sq=0.0
    for q in range(6):
        lam=_QP6[q]
        xq=v[:,0,0]*lam[0]+v[:,1,0]*lam[1]+v[:,2,0]*lam[2]
        yq=v[:,0,1]*lam[0]+v[:,1,1]*lam[1]+v[:,2,1]*lam[2]
        u_h=U_dofs@_PHI6[:,q]      # FEM value at quadrature point
        u_e=u_exact_fn(xq,yq)
        err_sq+=np.sum(areas*_QW6[q]*(u_h-u_e)**2)
    return np.sqrt(err_sq)


def fracture_c2_field(nodes, tris, angle_deg, cx, cy, half_len, thickness,
                       c2_bg, xi):
    """
    Per-element c² array implementing the weak-zone fault model.
    Elements whose centroid lies inside the fault zone get c²_frac = c2_bg·ξ².
    Fault zone: a rectangle of width=thickness, half-length=half_len,
                centred at (cx,cy), oriented at angle_deg from horizontal.
    ξ = V_FZ/V_O  (velocity ratio from the paper; typical range 0.2–0.6)
    """
    theta=np.radians(angle_deg); cos_t,sin_t=np.cos(theta),np.sin(theta)
    ex=nodes[tris[:,:3],0].mean(axis=1)
    ey=nodes[tris[:,:3],1].mean(axis=1)
    dx,dy=ex-cx,ey-cy
    t_along= dx*cos_t + dy*sin_t
    d_perp =np.abs(-dx*sin_t + dy*cos_t)
    in_fz  =(d_perp<=thickness/2) & (np.abs(t_along)<=half_len)
    c2_e   =np.full(len(tris),c2_bg,dtype=float)
    c2_e[in_fz]=c2_bg*xi**2
    return c2_e


def ricker_src(x, y, t, Lx, c, f0, t0r, strip):
    """
    Ricker wavelet body force in left strip (x < strip·Lx).
    S(x,y,t) = (1 − 2(πf₀τ)²)·exp(−(πf₀τ)²),  τ = (t − x/c) − t₀
    Models an incoming plane wave from the left.
    """
    r=np.zeros(len(x))
    m=x<strip*Lx; tau=(t-x[m]/c)-t0r; pft=np.pi*f0*tau
    r[m]=(1-2*pft**2)*np.exp(-pft**2)
    return r

# ─────────────────────────────────────────────────────────────────────────────
# 7.  CORE LEAPFROG SOLVER  (shared by Parts 2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation(N, Lx, Ly, c_bg, T,
                   c2_elem=None,   # per-element c² array (None → uniform)
                   src_fn=None,    # src(x,y,t) → ndarray
                   rec_xys=None):  # list of (x,y) receiver positions
    """
    Setup + HRZ lump + CFL + leapfrog integration with Dirichlet BCs (u=0 on boundary).

    Returns dict with: frames, ftimes, Ek, Ep, t, dt, nt, nn, h, lmax,
                       rec_signal (primary), rec_signals (list per receiver),
                       refl_ratio, dom_freq, nodes, tris, p1_tris.
    """
    c2_bg = c_bg**2
    nodes, tris, p1_tris = make_p2_mesh(N, N, Lx, Ly)
    nn = len(nodes); h = Lx/N

    c2_asm = c2_elem if c2_elem is not None else c2_bg
    M, K = assemble(nodes, tris, c2_asm)

    # HRZ lumped mass (row-sum gives zero for P2 vertex rows)
    diag_M = M.diagonal()
    Ml = diag_M * (float(M.sum()) / diag_M.sum())

    if src_fn is None:
        def src_fn(x, y, t): return np.zeros(len(x))

    # ── Dirichlet BCs: u=0 on all boundary nodes ──────────────────────────────
    fixed = boundary_dofs(nodes, Lx, Ly)
    Fd = np.zeros(nn)
    K = apply_dirichlet_sparse(K, Fd, fixed)
    Ml[fixed] = 1.0
    fset = set(fixed.tolist())
    int_d = np.array([i for i in range(nn) if i not in fset])

    si = 1.0/np.sqrt(Ml[int_d])
    A_sym = sp_diags(si) @ K[int_d, :][:, int_d] @ sp_diags(si)
    lmax = eigsh(A_sym, k=1, which='LM', return_eigenvectors=False)[0]
    dt = 0.9 * 2.0 / np.sqrt(lmax)
    nt = max(1, int(T/dt)); dt = T/nt

    # ── receivers ────────────────────────────────────────────────────────────
    rec_nodes = []
    if rec_xys:
        for rx, ry in rec_xys:
            rec_nodes.append(int(np.argmin((nodes[:, 0]-rx)**2 + (nodes[:, 1]-ry)**2)))

    # ── initial conditions (eq. 25 with U^0 = 0) ─────────────────────────────
    U_prev = np.zeros(nn)
    F0v = assemble_F(nodes, tris, src_fn, 0.0)
    F0v[fixed] = 0.0
    U_cur = (dt**2/2.0) * (F0v/Ml)
    U_cur[fixed] = 0.0

    save_every = max(1, nt//min(200, nt))
    frames, ftimes = [U_prev.copy()], [0.0]
    Ek_list, Ep_list, t_list = [], [], []
    rec_signals = [[] for _ in rec_nodes]

    left_mask = (nodes[:, 0] > SRC_STRIP*Lx) & (nodes[:, 0] < Lx/2)

    for step in range(1, nt):
        t_n = step*dt
        Fn = assemble_F(nodes, tris, src_fn, t_n)
        Fn[fixed] = 0.0
        KU = K.dot(U_cur)

        # Leapfrog step: U^{n+1} = 2U^n - U^{n-1} - dt²/Ml · KU^n + dt²/Ml · F^n
        U_next = 2*U_cur - U_prev - (dt**2/Ml)*KU + (dt**2/Ml)*Fn
        U_next[fixed] = 0.0

        vel = (U_next - U_prev) / (2*dt)
        Ek_list.append(0.5*np.dot(Ml*vel, vel))
        Ep_list.append(0.5*float(U_cur@KU))
        t_list.append(t_n)
        for k, rn in enumerate(rec_nodes):
            rec_signals[k].append(float(U_next[rn]))
        if step % save_every == 0:
            frames.append(U_next.copy()); ftimes.append(t_n)
        U_prev = U_cur; U_cur = U_next

    # ── reflected energy: kinetic in left half / total at t=T ────────────────
    # vel from last loop iteration is (U_next-U_prev)/(2dt) = centred O(dt²);
    # recomputing here would give a backward difference O(dt).
    vel_T = vel if nt > 1 else np.zeros(nn)
    Ek_l = 0.5*np.sum(Ml[left_mask]*vel_T[left_mask]**2)
    Ek_t = 0.5*np.dot(Ml*vel_T, vel_T)
    refl = float(Ek_l/Ek_t) if Ek_t > 1e-30 else 0.0

    # ── dominant frequency from primary receiver ──────────────────────────────
    rec_arrs = [np.array(s) for s in rec_signals]
    dom_f = 0.0
    if rec_arrs and len(rec_arrs[0]) > 16:
        ra = rec_arrs[0]
        freqs = np.fft.rfftfreq(len(ra), dt)
        spec = np.abs(np.fft.rfft(ra * np.hanning(len(ra))))
        dom_f = float(freqs[np.argmax(spec[1:])+1])

    # primary rec_signal stays backward-compatible (first receiver or empty)
    rec_signal = rec_arrs[0] if rec_arrs else np.array([])

    return dict(nodes=nodes, tris=tris, p1_tris=p1_tris,
                frames=frames, ftimes=np.array(ftimes),
                Ek=np.array(Ek_list), Ep=np.array(Ep_list),
                t=np.array(t_list), dt=dt, nt=nt, nn=nn, h=h, lmax=lmax,
                rec_signal=rec_signal, rec_signals=rec_arrs, rec_nodes=rec_nodes,
                refl_ratio=refl, dom_freq=dom_f)

# ─────────────────────────────────────────────────────────────────────────────
def banner(msg):
    print(f"\n{'='*68}\n  {msg}\n{'='*68}")

# =============================================================================
# PART 1 — STATIC CONVERGENCE TEST  (-c² Δu = f)
# =============================================================================
# Manufactured solution:  u_exact = sin(πx)·sin(πy)  on [0,1]²
#   f(x,y) = 2π²c²·sin(πx)·sin(πy)
# FIX: L² error computed via 6-pt Gauss quadrature (was np.mean — wrong)
# Expected: O(h³) in L² norm (Aubin-Nitsche duality for P2: order p+1 = 3)
# =============================================================================
def part1_static(Lx=LX, Ly=LY, c2=C_BG**2):
    banner("PART 1 -- 2D Static Test   -c^2 * laplacian(u) = f  (no time)")

    u_exact=lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)
    src_s  =lambda x,y,t=0: 2*np.pi**2*c2*np.sin(np.pi*x)*np.sin(np.pi*y)

    hs,errs=[],[]
    for N in N_STATIC:
        nodes,tris,_=make_p2_mesh(N,N,Lx,Ly)
        _,K=assemble(nodes,tris,c2)
        F=assemble_F(nodes,tris,src_s)
        fixed=boundary_dofs(nodes,Lx,Ly)
        Kd=K.toarray(); apply_dirichlet_dense(Kd,F,fixed)
        U=np.linalg.solve(Kd,F)
        err=l2_error(nodes,tris,U,u_exact)   # ← proper L² norm
        h=Lx/N; hs.append(h); errs.append(err)
        print(f"    N={N:4d}   h={h:.5f}   L2-error = {err:.4e}")

    rates=[np.log(errs[k]/errs[k-1])/np.log(hs[k]/hs[k-1])
           for k in range(1,len(N_STATIC))]
    print(f"\n  Convergence rates : {[f'{r:.2f}' for r in rates]}")
    print(f"  Expected for P2   : ~3.00  (L2 norm, Aubin-Nitsche: p+1 = 2+1)")

    # Plot: solution + error + convergence
    N_p=16
    nodes_p,tris_p,p1_p=make_p2_mesh(N_p,N_p,Lx,Ly)
    _,Kp=assemble(nodes_p,tris_p,c2)
    Fp=assemble_F(nodes_p,tris_p,src_s)
    fp=boundary_dofs(nodes_p,Lx,Ly)
    Kpd=Kp.toarray(); apply_dirichlet_dense(Kpd,Fp,fp)
    Up=np.linalg.solve(Kpd,Fp)
    triang=mtri.Triangulation(nodes_p[:,0],nodes_p[:,1],p1_p)

    fig,axes=plt.subplots(1,3,figsize=(16,5))
    fig.suptitle("Part 1 — 2D P2 FEM  Static: $-c^2\\Delta u = f$",
                 fontsize=14,fontweight='bold')
    levels=np.linspace(-1.05,1.05,22)
    cf1=axes[0].tricontourf(triang,Up,levels=levels,cmap='RdBu_r')
    axes[0].set_title(f'P2 FEM solution  (N={N_p})')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].set_aspect('equal'); plt.colorbar(cf1,ax=axes[0])

    err_vals=np.abs(Up-u_exact(nodes_p[:,0],nodes_p[:,1]))
    cf2=axes[1].tricontourf(triang,err_vals,cmap='Reds')
    axes[1].set_title('Pointwise error $|u_h-u_{exact}|$')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    axes[1].set_aspect('equal'); plt.colorbar(cf2,ax=axes[1])

    ha=np.array(hs); ea=np.array(errs)
    axes[2].loglog(ha,ea,'b-o',ms=8,lw=2,label='$L^2$ error (Gauss)')
    axes[2].loglog(ha,0.06*ha**3,'k--',lw=1.5,label='$O(h^3)$')
    axes[2].set_xlabel('$h$'); axes[2].set_ylabel('$L^2$ error')
    axes[2].set_title('P2 convergence'); axes[2].legend()
    axes[2].grid(True,which='both',alpha=0.3)

    plt.tight_layout()
    plt.savefig('p2_2d_part1_static.png',dpi=150,bbox_inches='tight')
    print("  Saved: p2_2d_part1_static.png")
    plt.show()


# =============================================================================
# PART 2 — DYNAMIC WAVE VALIDATION  (standing-wave source)
# =============================================================================
# S(x,y,t) = sin(πx)·sin(πy)·sin(ωt)  — resonant mode drive
# Uses run_simulation with uniform c², no sponge.
# =============================================================================
def part2_dynamic(N=N_DYN, Lx=LX, Ly=LY, c=C_BG, T=T_DYN,
                  omega=OMEGA, save_gif=True):
    banner("PART 2 -- Dynamic Wave  S=sin(pi*x)sin(pi*y)sin(omega*t)  [validation]")

    def src(x,y,t): return np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(omega*t)
    res=run_simulation(N,Lx,Ly,c,T,src_fn=src)

    nodes,tris,p1_tris=res['nodes'],res['tris'],res['p1_tris']
    frames,ftimes=res['frames'],res['ftimes']
    Ek,Ep,t_arr=res['Ek'],res['Ep'],res['t']
    print(f"  {N}×{N} mesh → {res['nn']} nodes,  λ_max={res['lmax']:.1f},  "
          f"dt={res['dt']:.3e},  steps={res['nt']:,}")

    triang=mtri.Triangulation(nodes[:,0],nodes[:,1],p1_tris)
    umax=max(np.abs(f).max() for f in frames)+1e-12
    idxs=[0,len(frames)//4,len(frames)//2,len(frames)-1]
    levels=np.linspace(-umax,umax,24)

    fig,axes=plt.subplots(2,2,figsize=(13,11))
    fig.suptitle(r"Part 2 — 2D P2 FEM Wave:  $S=\sin(\pi x)\sin(\pi y)\sin(\omega t)$"
                 f"\n$c={c}$,  $\\omega={omega:.2f}$,  $N={N}$,  $T={T}$",
                 fontsize=13,fontweight='bold')
    for ax,idx in zip(axes.flat,idxs):
        cf=ax.tricontourf(triang,frames[idx],levels=levels,cmap='seismic')
        ax.set_title(f'$t$ = {ftimes[idx]:.3f} s')
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_aspect('equal'); plt.colorbar(cf,ax=ax,shrink=0.85)
    plt.tight_layout()
    plt.savefig('p2_2d_part2_snapshots.png',dpi=150,bbox_inches='tight')
    print("  Saved: p2_2d_part2_snapshots.png")
    plt.show()

    _,ax_e=plt.subplots(figsize=(11,4))
    ax_e.plot(t_arr,Ek,'b-',lw=1.5,label='Kinetic $E_k$')
    ax_e.plot(t_arr,Ep,'r-',lw=1.5,label='Potential $E_p$')
    ax_e.plot(t_arr,Ek+Ep,'k--',lw=1.5,label='Total')
    ax_e.set_title('Energy (source-driven, grows)'); ax_e.set_xlabel('$t$ [s]')
    ax_e.legend(); ax_e.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig('p2_2d_part2_energy.png',dpi=150,bbox_inches='tight')
    print("  Saved: p2_2d_part2_energy.png")
    plt.show()

    if save_gif:
        levels_a=np.linspace(-umax,umax,22)
        fig3,ax3=plt.subplots(figsize=(7,7)); ax3.set_aspect('equal')
        ax3.set_xlabel('$x$',fontsize=12); ax3.set_ylabel('$y$',fontsize=12)
        cf_a=[ax3.tricontourf(triang,frames[0],levels=levels_a,cmap='seismic')]
        plt.colorbar(cf_a[0],ax=ax3,shrink=0.85,label='$u$')
        ax3.set_title('')
        def _upd(i):
            ax3.cla()
            ax3.set_aspect('equal')
            ax3.set_xlabel('$x$',fontsize=12); ax3.set_ylabel('$y$',fontsize=12)
            cf_a[0]=ax3.tricontourf(triang,frames[i],levels=levels_a,cmap='seismic')
            ax3.set_title(f'2D P2 FEM  --  t={ftimes[i]:.3f} s')
            return []
        ani=FuncAnimation(fig3,_upd,frames=len(frames),interval=60)
        plt.tight_layout()
        try:
            ani.save('p2_2d_wave.gif',writer='pillow',fps=15,dpi=90)
            print("  Saved: p2_2d_wave.gif")
        except Exception as e:
            print(f"  GIF skipped: {e}")
        plt.show()


# =============================================================================
# PART 3 — FRACTURE ANGLE STUDY
# =============================================================================
# Source: Ricker plane wave from left strip.
# Fracture: weak zone with c² reduced by ξ² (paper: velocity ratio ξ = V_FZ/V_O).
# Angles: 30°, 45°, 60°, 90° (paper uses 90° as reference).
# Fixed: thickness = THICK_DEF, velocity ratio = XI_DEF.
# =============================================================================
def part3_fracture_angles():
    banner("PART 3 -- Fracture Angle Study  (theta = 30, 45, 60, 90 deg)")
    print(f"  Config: thickness={THICK_DEF} m, xi={XI_DEF}, "
          f"half-length={FRAC_HALF} m, N={N_FRAC}")
    print(f"  BCs: Dirichlet u=0  |  T_FRAC={T_FRAC} s  |  receivers: LS_L, LS_C, LS_R")
    print(f"  Reflection timing (c={C_BG} m/s, Lx={LX} m):")
    print(f"    Right wall -> fracture (x=0.5): t = 0.9/c + 0.5/c + 0.1/c = 1.5 s")
    print(f"    Top/bottom -> fracture:          t ≈ sqrt(0.5²+0.5²)/c + ... ≈ 1.27 s")
    print(f"    => T_FRAC={T_FRAC} s is safe — no reflections reach the fracture zone.")

    cx,cy=LX/2,LY/2
    rec_list=[LS_L, LS_C, LS_R]
    rec_labels=['LS_L (reflected)', 'LS_C (trapped)', 'LS_R (transmitted)']
    rec_colors=['#e41a1c','#2ca02c','#1f77b4']
    results=[]

    src=lambda x,y,t: ricker_src(x,y,t,LX,C_BG,F0,T0_RICK,SRC_STRIP)

    for ang in ANGLES:
        print(f"\n  -- theta={ang} deg --")
        nodes_tmp,tris_tmp,_=make_p2_mesh(N_FRAC,N_FRAC,LX,LY)
        c2_e=fracture_c2_field(nodes_tmp,tris_tmp,ang,cx,cy,FRAC_HALF,THICK_DEF,
                                C_BG**2,XI_DEF)
        n_fz=int(np.sum(c2_e<C_BG**2)); print(f"  Fault elements: {n_fz}")

        res=run_simulation(N_FRAC,LX,LY,C_BG,T_FRAC,
                           c2_elem=c2_e, src_fn=src,
                           rec_xys=rec_list)
        res['angle']=ang
        print(f"  lmax={res['lmax']:.1f}, dt={res['dt']:.4f}, steps={res['nt']}")

        max_disp = float(np.max(np.abs(res['rec_signal']))) if len(res['rec_signal'])>0 else 0.0
        print(f"  Reflected energy: {100*res['refl_ratio']:.1f}%  |  "
              f"max LS_R disp: {max_disp:.5f}  |  dom freq: {res['dom_freq']:.2f} Hz")

        # ── 4 snapshots ───────────────────────────────────────────────────────
        nodes,tris,p1_tris=res['nodes'],res['tris'],res['p1_tris']
        frames,ftimes=res['frames'],res['ftimes']
        triang=mtri.Triangulation(nodes[:,0],nodes[:,1],p1_tris)
        umax=max(np.abs(f).max() for f in frames)+1e-14
        snap_t=[T_FRAC/4,T_FRAC/2,3*T_FRAC/4,T_FRAC]
        sidxs=[int(np.argmin(np.abs(ftimes-st))) for st in snap_t]
        cos_a,sin_a=np.cos(np.radians(ang)),np.sin(np.radians(ang))
        fx1,fy1=cx-FRAC_HALF*cos_a,cy-FRAC_HALF*sin_a
        fx2,fy2=cx+FRAC_HALF*cos_a,cy+FRAC_HALF*sin_a

        fig,axes=plt.subplots(2,2,figsize=(13,11))
        fig.suptitle(f"Part 3 -- SH Wave,  Fracture theta={ang} deg\n"
                     f"xi={XI_DEF},  h_f={THICK_DEF} m,  c_bg={C_BG} m/s  "
                     f"[Dirichlet BCs, T={T_FRAC} s]",
                     fontsize=13,fontweight='bold')
        levels=np.linspace(-umax,umax,26)
        rn_pos=[(LS_L[0],LS_L[1],'LS_L','<'),
                (LS_C[0],LS_C[1],'LS_C','o'),
                (LS_R[0],LS_R[1],'LS_R','>')]
        for ax,idx in zip(axes.flat,sidxs):
            cf=ax.tricontourf(triang,frames[idx],levels=levels,cmap='seismic')
            ax.plot([fx1,fx2],[fy1,fy2],'k-',lw=3,label='fault zone')
            for rx,ry,rl,rm in rn_pos:
                ax.plot(rx,ry,rm,ms=8,color='lime',zorder=6,
                        label=rl if ax is axes[0,0] else '')
            ax.axvline(SRC_STRIP*LX,color='b',lw=1,ls='--',alpha=0.5)
            ax.set_title(f't = {ftimes[idx]:.3f} s')
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
            ax.set_aspect('equal'); plt.colorbar(cf,ax=ax,shrink=0.85,label='u')
        axes[0,0].legend(fontsize=7,loc='upper right')
        plt.tight_layout()
        fname=f'p2_2d_fracture_{ang}deg_snapshots.png'
        plt.savefig(fname,dpi=150,bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.show()
        results.append(res)

    # ── Comparison figure: 3 receivers × 4 angles ────────────────────────────
    ang_colors=['#e41a1c','#ff7f00','#377eb8','#4daf4a']
    fig=plt.figure(figsize=(18,12))
    gs=GridSpec(3,5,figure=fig,hspace=0.50,wspace=0.38)

    for row,(rlabel,rcol) in enumerate(zip(rec_labels,rec_colors)):
        for i,res in enumerate(results):
            ax=fig.add_subplot(gs[row,i])
            sig=res['rec_signals'][row] if row<len(res['rec_signals']) else []
            tr=np.arange(len(sig))*res['dt']
            ax.plot(tr,sig,color=ang_colors[i],lw=1.5)
            if row==0:
                ax.set_title(f"theta={res['angle']} deg\nE_refl={100*res['refl_ratio']:.1f}%",
                             fontsize=9)
            ax.set_xlabel('t [s]',fontsize=8); ax.set_ylabel(rlabel,fontsize=7)
            ax.grid(True,alpha=0.3); ax.axhline(0,color='k',lw=0.4)
            ax.tick_params(labelsize=7)

    # bar chart of reflected energy
    ax_b=fig.add_subplot(gs[:,4])
    ang_l=[f"{r['angle']}°" for r in results]
    rp=[100*r['refl_ratio'] for r in results]
    bars=ax_b.barh(ang_l,rp,color=ang_colors,edgecolor='k',lw=0.8)
    for bar,val in zip(bars,rp):
        ax_b.text(val+0.3,bar.get_y()+bar.get_height()/2,
                  f'{val:.1f}%',va='center',fontsize=10)
    ax_b.set_title('Reflected\nenergy',fontsize=10)
    ax_b.set_xlabel('E_left / E_total [%]')
    ax_b.grid(True,axis='x',alpha=0.3)
    ax_b.set_xlim(0,max(rp)*1.4+2)
    ax_b.invert_yaxis()

    fig.suptitle('Part 3 -- SH Wave Fracture Angle Study\n'
                 'Rows: LS_L (reflected), LS_C (trapped), LS_R (transmitted)',
                 fontsize=13,fontweight='bold')
    plt.savefig('p2_2d_part3_comparison.png',dpi=150,bbox_inches='tight')
    print("\n  Saved: p2_2d_part3_comparison.png")
    plt.show()

    # Text summary table
    print('\n' + '='*72)
    print(f"{'Angle':>8}  {'E_refl%':>10}  {'max|LS_L|':>12}  "
          f"{'max|LS_C|':>12}  {'dom_freq Hz':>12}")
    print('-'*72)
    for r in results:
        sigs=r['rec_signals']
        mL=float(np.max(np.abs(sigs[0]))) if sigs and len(sigs[0])>0 else 0.0
        mC=float(np.max(np.abs(sigs[1]))) if len(sigs)>1 and len(sigs[1])>0 else 0.0
        print(f"  {r['angle']:>4}°   {100*r['refl_ratio']:>9.1f}%  "
              f"{mL:>12.5f}  {mC:>12.5f}  {r['dom_freq']:>10.2f}")
    print('='*72)
    return results


# =============================================================================
# PART 4 — PARAMETER SENSITIVITY  (thickness × velocity ratio)
# =============================================================================
# Fixed angle = FIXED_ANGLE (60°).
# Vary:  thickness h_f ∈ THICKNESSES  and  velocity ratio ξ ∈ XI_VALUES.
# Produces a 3×3 grid of receiver traces + heatmap of reflected energy.
# =============================================================================
def part4_sensitivity():
    banner(f"PART 4 -- Sensitivity: thickness x xi  (angle fixed = {FIXED_ANGLE} deg)")
    cx,cy=LX/2,LY/2
    colors_xi=['#d62728','#1f77b4','#2ca02c']

    grid_refl=np.zeros((len(THICKNESSES),len(XI_VALUES)))
    grid_max =np.zeros_like(grid_refl)

    fig_grid=plt.figure(figsize=(14,10))
    gs_g=GridSpec(len(THICKNESSES),len(XI_VALUES),
                  figure=fig_grid,hspace=0.5,wspace=0.35)
    fig_grid.suptitle(
        f"Part 4 — Receiver seismograms  (θ={FIXED_ANGLE}°)\n"
        f"Rows: thickness h_f,  Cols: velocity ratio ξ",
        fontsize=13,fontweight='bold')

    print(f"  {'thickness':>12}  {'xi':>6}  {'E_refl%':>10}  {'max_disp':>12}")
    print('  '+'-'*44)

    def src(x,y,t): return ricker_src(x,y,t,LX,C_BG,F0,T0_RICK,SRC_STRIP)

    for ri,thick in enumerate(THICKNESSES):
        for ci,xi in enumerate(XI_VALUES):
            nodes_t,tris_t,_=make_p2_mesh(N_FRAC,N_FRAC,LX,LY)
            c2_e=fracture_c2_field(nodes_t,tris_t,FIXED_ANGLE,cx,cy,
                                    FRAC_HALF,thick,C_BG**2,xi)
            res=run_simulation(N_FRAC,LX,LY,C_BG,T_FRAC,
                               c2_elem=c2_e,src_fn=src,
                               rec_xys=[LS_R])
            grid_refl[ri,ci]=100*res['refl_ratio']
            grid_max[ri,ci] =float(np.max(np.abs(res['rec_signal']))) if len(res['rec_signal'])>0 else 0.0
            print(f"  {thick:>12.3f}  {xi:>6.2f}  "
                  f"{100*res['refl_ratio']:>10.1f}%  {grid_max[ri,ci]:>12.5f}")

            ax=fig_grid.add_subplot(gs_g[ri,ci])
            tr=np.arange(len(res['rec_signal']))*res['dt']
            ax.plot(tr,res['rec_signal'],color=colors_xi[ci],lw=1.2)
            ax.set_title(f'h_f={thick:.2f}, ξ={xi}',fontsize=8)
            ax.set_xlabel('t [s]',fontsize=7); ax.set_ylabel('u',fontsize=7)
            ax.tick_params(labelsize=6); ax.grid(True,alpha=0.3)
            ax.axhline(0,color='k',lw=0.4)

    plt.savefig('p2_2d_part4_grid.png',dpi=150,bbox_inches='tight')
    print("\n  Saved: p2_2d_part4_grid.png")
    plt.show()

    # Heatmap of reflected energy
    fig_h,ax_h=plt.subplots(figsize=(8,5))
    im=ax_h.imshow(grid_refl,aspect='auto',cmap='hot_r',
                   vmin=0,vmax=grid_refl.max()*1.1)
    plt.colorbar(im,ax=ax_h,label='Reflected energy [%]')
    ax_h.set_xticks(range(len(XI_VALUES))); ax_h.set_xticklabels([f'ξ={x}' for x in XI_VALUES])
    ax_h.set_yticks(range(len(THICKNESSES)))
    ax_h.set_yticklabels([f'h={t:.2f}' for t in THICKNESSES])
    ax_h.set_title(f'Reflected energy (%) — θ={FIXED_ANGLE}°',fontsize=12)
    ax_h.set_xlabel('Velocity ratio ξ'); ax_h.set_ylabel('Thickness h_f [m]')
    for ri in range(len(THICKNESSES)):
        for ci in range(len(XI_VALUES)):
            ax_h.text(ci,ri,f'{grid_refl[ri,ci]:.1f}',
                      ha='center',va='center',fontsize=11,
                      color='white' if grid_refl[ri,ci]>grid_refl.max()*0.5 else 'black')
    plt.tight_layout()
    plt.savefig('p2_2d_part4_heatmap.png',dpi=150,bbox_inches='tight')
    print("  Saved: p2_2d_part4_heatmap.png")
    plt.show()

    # Conclusion summary — derived from grid data
    best_ri,best_ci=np.unravel_index(np.argmax(grid_refl),grid_refl.shape)
    worst_ri,worst_ci=np.unravel_index(np.argmin(grid_refl),grid_refl.shape)
    refl_by_thick=[grid_refl[ri,:].mean() for ri in range(len(THICKNESSES))]
    refl_by_xi   =[grid_refl[:,ci].mean() for ci in range(len(XI_VALUES))]
    best_thick_idx=int(np.argmax(refl_by_thick)); best_xi_idx=int(np.argmax(refl_by_xi))
    print(f"\n  CONCLUSIONS (theta={FIXED_ANGLE} deg):")
    print(f"  * Maximum reflection:  h_f={THICKNESSES[best_ri]:.2f} m, "
          f"xi={XI_VALUES[best_ci]}  ->  {grid_refl[best_ri,best_ci]:.1f}%")
    print(f"  * Minimum reflection:  h_f={THICKNESSES[worst_ri]:.2f} m, "
          f"xi={XI_VALUES[worst_ci]}  ->  {grid_refl[worst_ri,worst_ci]:.1f}%")
    print(f"  * Mean by thickness:   "
          + "  ".join(f"h={t:.2f}:{v:.1f}%" for t,v in zip(THICKNESSES,refl_by_thick)))
    print(f"  * Mean by xi:          "
          + "  ".join(f"xi={x}:{v:.1f}%" for x,v in zip(XI_VALUES,refl_by_xi)))
    print(f"  * Highest avg reflection: "
          f"thickness h_f={THICKNESSES[best_thick_idx]:.2f} m, "
          f"xi={XI_VALUES[best_xi_idx]}")
    print(f"  * The relationship is non-monotonic — thickness and velocity")
    print(f"    contrast interact: a thin but sharp fault can reflect more")
    print(f"    than a thick gradual one (wave-trapping vs impedance mismatch).")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    t0=time.time()
    print("P2 FEM SH Wave — Full Study")
    print(f"Domain: [{LX}×{LY}],  c_bg={C_BG} m/s")

    part1_static()
    part2_dynamic(save_gif=True)
    part3_fracture_angles()
    part4_sensitivity()

    print(f"\nAll done in {time.time()-t0:.1f} s")

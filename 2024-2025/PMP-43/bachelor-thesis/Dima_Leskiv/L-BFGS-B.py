"""
SIR-модель з оптимальним контролем, SIRV та «звичайною» SIR.
Нові параметри роблять різницю між контроль/без-контролю помітною,
а u(t) показується шматково-сталою (1-денні блоки).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- глобальні константи ----------
T, dt = 30.0, 0.1
N_steps = int(T/dt)
t = np.linspace(0, T, N_steps + 1)
N = 1000.0

# ---------- (A) оптимальний контроль SIR ----------
beta, delta, mu, lam = 1.0, 0.2, 1e-4, 1e-4
a, b     = 10.0, 0.1          # *сильно* караємо I, дешево колемо
u_max    = 0.30               # до 30 %/день
S0, I0, R0 = 950, 50, 0       # стартові

def fwd_rhs(S, I, R, u):
    dS = lam - beta*S*I/N - mu*S - u*S
    dI = beta*S*I/N - (delta+mu)*I
    dR = delta*I - mu*R + u*S
    return np.array([dS, dI, dR])

def adj_rhs(lS, lI, lR, S, I, R, u):
    dlS = lS*(beta*I/N + mu + u) - lI*(beta*I/N) - lR*u
    dlI = -a + lS*(beta*S/N) - lI*(beta*S/N - delta - mu) - lR*delta
    dlR = lR*mu
    return np.array([dlS, dlI, dlR])

def rk4_forward(u):
    S = np.zeros_like(t); I = np.zeros_like(t); R = np.zeros_like(t)
    S[0], I[0], R[0] = S0, I0, R0
    for k in range(N_steps):
        y  = np.array([S[k], I[k], R[k]])
        k1 = fwd_rhs(*y,               u[k])
        k2 = fwd_rhs(*(y+0.5*dt*k1),   u[k])
        k3 = fwd_rhs(*(y+0.5*dt*k2),   u[k])
        k4 = fwd_rhs(*(y+dt*k3),       u[k+1])
        S[k+1], I[k+1], R[k+1] = y + (dt/6)*(k1+2*k2+2*k3+k4)
    return S, I, R

def rk4_backward(S,I,R,u):
    lS = np.zeros_like(t); lI = np.zeros_like(t); lR = np.zeros_like(t)
    for k in range(N_steps,0,-1):
        y  = np.array([lS[k], lI[k], lR[k]])
        k1 = adj_rhs(*y,  S[k],I[k],R[k],u[k])
        k2 = adj_rhs(*(y-0.5*dt*k1), S[k-1],I[k-1],R[k-1],u[k-1])
        k3 = adj_rhs(*(y-0.5*dt*k2), S[k-1],I[k-1],R[k-1],u[k-1])
        k4 = adj_rhs(*(y-dt*k3),     S[k-1],I[k-1],R[k-1],u[k-1])
        lS[k-1], lI[k-1], lR[k-1] = y - (dt/6)*(k1+2*k2+2*k3+k4)
    return lS, lI, lR

def update_u(S,lS,lR):             # формула з Понтрягіна
    return np.clip((S*(lS-lR))/b, 0.0, u_max)

u = np.zeros_like(t)
for _ in range(80):                # forward–backward sweep
    S_opt, I_opt, R_opt = rk4_forward(u)
    lS,lI,lR            = rk4_backward(S_opt,I_opt,R_opt,u)
    u_new               = update_u(S_opt,lS,lR)
    if np.linalg.norm(u_new-u,np.inf) < 1e-4:
        u = u_new
        S_opt,I_opt,R_opt = rk4_forward(u)   # фінальний прогін
        break
    u = u_new

# --- перетворюємо u(t) у «сходинки» по 1 дню ---
block = int(1/dt)                  # 10 кроків = 1 день
n_full = (N_steps+1)//block        # кількість повних блоків
u_blocks = u[:n_full*block].reshape(n_full, block)
u_step = np.repeat(u_blocks.mean(axis=1), block)
u_step = np.append(u_step, u[n_full*block:]) # хвіст (0.0-0.1 дн.)

S_opt,I_opt,R_opt = rk4_forward(u_step)      # траєкторії під «сходинки»

# ---------- (B) SIRV без оптимізації ----------
λ, βv, μv, δv = 1e-4, 0.6, 1e-4, 0.2
θ, η, ζ, ξ, τ = 0.2, 0.05, 0.01, 0.1, 0.01
S2 = np.zeros_like(t); V2 = np.zeros_like(t); I2 = np.zeros_like(t); R2 = np.zeros_like(t)
S2[0], I2[0], R2[0] = 950, 50, 0
def dS2(s,v,i): return (1-θ)*λ*N - βv*s*i/N + ζ*v - η*s - μv*s
def dV2(s,v,i): return θ*λ*N + η*s - ξ*βv*v*i/N - (ζ+μv)*v
def dI2(s,v,i): return βv*s*i/N + ξ*βv*v*i/N - (τ+δv+μv)*i
def dR2(i,r):    return δv*i - μv*r
for k in range(N_steps):
    k1_S=dS2(S2[k],V2[k],I2[k]); k1_V=dV2(S2[k],V2[k],I2[k])
    k1_I=dI2(S2[k],V2[k],I2[k]); k1_R=dR2(I2[k],R2[k])
    k2_S=dS2(S2[k]+0.5*dt*k1_S, V2[k]+0.5*dt*k1_V, I2[k]+0.5*dt*k1_I)
    k2_V=dV2(S2[k]+0.5*dt*k1_S, V2[k]+0.5*dt*k1_V, I2[k]+0.5*dt*k1_I)
    k2_I=dI2(S2[k]+0.5*dt*k1_S, V2[k]+0.5*dt*k1_V, I2[k]+0.5*dt*k1_I)
    k2_R=dR2(I2[k]+0.5*dt*k1_I, R2[k]+0.5*dt*k1_R)
    k3_S=dS2(S2[k]+0.5*dt*k2_S, V2[k]+0.5*dt*k2_V, I2[k]+0.5*dt*k2_I)
    k3_V=dV2(S2[k]+0.5*dt*k2_S, V2[k]+0.5*dt*k2_V, I2[k]+0.5*dt*k2_I)
    k3_I=dI2(S2[k]+0.5*dt*k2_S, V2[k]+0.5*dt*k2_V, I2[k]+0.5*dt*k2_I)
    k3_R=dR2(I2[k]+0.5*dt*k2_I, R2[k]+0.5*dt*k2_R)
    k4_S=dS2(S2[k]+dt*k3_S, V2[k]+dt*k3_V, I2[k]+dt*k3_I)
    k4_V=dV2(S2[k]+dt*k3_S, V2[k]+dt*k3_V, I2[k]+dt*k3_I)
    k4_I=dI2(S2[k]+dt*k3_S, V2[k]+dt*k3_V, I2[k]+dt*k3_I)
    k4_R=dR2(I2[k]+dt*k3_I, R2[k]+dt*k3_R)
    S2[k+1]=S2[k]+(dt/6)*(k1_S+2*k2_S+2*k3_S+k4_S)
    V2[k+1]=V2[k]+(dt/6)*(k1_V+2*k2_V+2*k3_V+k4_V)
    I2[k+1]=I2[k]+(dt/6)*(k1_I+2*k2_I+2*k3_I+k4_I)
    R2[k+1]=R2[k]+(dt/6)*(k1_R+2*k2_R+2*k3_R+k4_R)

# ---------- (C) «звичайна» SIR ----------
S3 = np.zeros_like(t); I3 = np.zeros_like(t); R3 = np.zeros_like(t)
S3[0], I3[0], R3[0] = 950, 50, 0
def dS3(s,i): return lam*N - beta*s*i/N - mu*s
def dI3(s,i): return beta*s*i/N - delta*i - mu*i
def dR3(i,r): return delta*i - mu*r
for k in range(N_steps):
    k1S=dS3(S3[k],I3[k]); k1I=dI3(S3[k],I3[k]); k1R=dR3(I3[k],R3[k])
    k2S=dS3(S3[k]+0.5*dt*k1S, I3[k]+0.5*dt*k1I)
    k2I=dI3(S3[k]+0.5*dt*k1S, I3[k]+0.5*dt*k1I)
    k2R=dR3(I3[k]+0.5*dt*k1I, R3[k]+0.5*dt*k1R)
    k3S=dS3(S3[k]+0.5*dt*k2S, I3[k]+0.5*dt*k2I)
    k3I=dI3(S3[k]+0.5*dt*k2S, I3[k]+0.5*dt*k2I)
    k3R=dR3(I3[k]+0.5*dt*k2I, R3[k]+0.5*dt*k2R)
    k4S=dS3(S3[k]+dt*k3S, I3[k]+dt*k3I)
    k4I=dI3(S3[k]+dt*k3S, I3[k]+dt*k3I)
    k4R=dR3(I3[k]+dt*k3I, R3[k]+dt*k3R)
    S3[k+1]=S3[k]+(dt/6)*(k1S+2*k2S+2*k3S+k4S)
    I3[k+1]=I3[k]+(dt/6)*(k1I+2*k2I+2*k3I+k4I)
    R3[k+1]=R3[k]+(dt/6)*(k1R+2*k2R+2*k3R+k4R)

# ---------- (D) графіки ----------
os.makedirs("figures", exist_ok=True)

# u(t) – сходинки
plt.figure(figsize=(6,2.5))
plt.step(t, u_step, where="mid", lw=2, color="purple")
plt.xlabel("Дні"); plt.ylabel("u (частка щеплених/день)")
plt.title("Оптимальна інтенсивність вакцинації u(t)")
plt.ylim(-0.01, u_max*1.05); plt.grid(axis="y")
plt.tight_layout(); plt.savefig("figures/optimal_vaccination_rate.png", dpi=300); plt.close()

# порівняння S, I, R
fig, ax = plt.subplots(1,3, figsize=(13,3), sharex=False)

ax[0].plot(t, S_opt,   label="S (opt)",  lw=2)
ax[0].plot(t, S2,      label="S (SIRV)", lw=1.8)
ax[0].plot(t, S3, '--',label="S (SIR)",  lw=1.5)
ax[0].set_title("S(t)"); ax[0].set_xlabel("Дні"); ax[0].set_ylabel("Осіб"); ax[0].grid(); ax[0].legend()

ax[1].plot(t, I_opt,   label="I (opt)",  lw=2)
ax[1].plot(t, I2,      label="I (SIRV)", lw=1.8)
ax[1].plot(t, I3, '--',label="I (SIR)",  lw=1.5)
ax[1].set_title("I(t)"); ax[1].set_xlabel("Дні"); ax[1].grid(); ax[1].legend()

ax[2].plot(t, R_opt,   label="R (opt)",  lw=2)
ax[2].plot(t, R2,      label="R (SIRV)", lw=1.8)
ax[2].plot(t, R3, '--',label="R (SIR)",  lw=1.5)
ax[2].set_title("R(t)"); ax[2].set_xlabel("Дні"); ax[2].grid(); ax[2].legend()

plt.tight_layout()
plt.savefig("figures/sir_vs_sirv_vs_sir_comparison.png", dpi=300)
plt.close()

print("✅ Файли збережено у папці ./figures")

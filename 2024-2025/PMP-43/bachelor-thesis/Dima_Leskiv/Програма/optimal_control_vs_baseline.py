
import numpy as np
import matplotlib.pyplot as plt
import os

# ===== 0. глобальні налаштування ============================================
T, dt = 30.0, 0.1
N_steps = int(T / dt)
t = np.linspace(0, T, N_steps + 1)
N = 1000.0
os.makedirs("figures", exist_ok=True)

# ===== (A) SIR з оптимальним контролем ======================================
beta, delta, mu, lam = 1.0, 0.2, 1e-4, 1e-4
a, b   = 10.0, 0.1          # ваги функціонала: сильна «кара» за I, дешева вакцина
u_max  = 0.30               # до 30 % сприйнятливих на добу
S0, I0, R0 = 950, 50, 0

def fwd_rhs(S, I, R, u):
    return np.array([
        lam - beta*S*I/N - mu*S - u*S,
        beta*S*I/N - (delta + mu)*I,
        delta*I - mu*R + u*S
    ])

def adj_rhs(lS, lI, lR, S, I, R, u):
    return np.array([
        lS*(beta*I/N + mu + u) - lI*(beta*I/N) - lR*u,
        -a + lS*(beta*S/N) - lI*(beta*S/N - delta - mu) - lR*delta,
        lR*mu
    ])

def rk4_forward(u):
    S = np.zeros_like(t); I = np.zeros_like(t); R = np.zeros_like(t)
    S[0], I[0], R[0] = S0, I0, R0
    for k in range(N_steps):
        y  = np.array([S[k], I[k], R[k]])
        k1 = fwd_rhs(*y,             u[k])
        k2 = fwd_rhs(*(y + 0.5*dt*k1), u[k])
        k3 = fwd_rhs(*(y + 0.5*dt*k2), u[k])
        k4 = fwd_rhs(*(y + dt*k3),     u[k+1])
        S[k+1], I[k+1], R[k+1] = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return S, I, R

def rk4_backward(S, I, R, u):
    lS = np.zeros_like(t); lI = np.zeros_like(t); lR = np.zeros_like(t)
    for k in range(N_steps, 0, -1):
        y  = np.array([lS[k], lI[k], lR[k]])
        k1 = adj_rhs(*y, S[k],   I[k],   R[k],   u[k])
        k2 = adj_rhs(*(y - 0.5*dt*k1), S[k-1], I[k-1], R[k-1], u[k-1])
        k3 = adj_rhs(*(y - 0.5*dt*k2), S[k-1], I[k-1], R[k-1], u[k-1])
        k4 = adj_rhs(*(y - dt*k3),     S[k-1], I[k-1], R[k-1], u[k-1])
        lS[k-1], lI[k-1], lR[k-1] = y - (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return lS, lI, lR

def update_u(S, lS, lR):
    return np.clip((S * (lS - lR)) / b, 0.0, u_max)

u = np.zeros_like(t)
for _ in range(80):                    # forward–backward sweep
    S_opt, I_opt, R_opt = rk4_forward(u)
    lS, lI, lR          = rk4_backward(S_opt, I_opt, R_opt, u)
    u_new               = update_u(S_opt, lS, lR)
    if np.linalg.norm(u_new - u, np.inf) < 1e-4:
        u = u_new
        S_opt, I_opt, R_opt = rk4_forward(u)  # фінальний прогін
        break
    u = u_new

# --- зробимо u(t) шматково-сталим по 1 дню (сходинки) ---
block = int(1 / dt)                    # 10 кроків = 1 доба
u_blocks = u[:(N_steps // block) * block].reshape(-1, block)
u_step = np.repeat(u_blocks.mean(axis=1), block)
u_step = np.append(u_step, u[len(u_step):])   # хвіст <1 доби
S_opt, I_opt, R_opt = rk4_forward(u_step)     # траєкторії під «сходинки»

# ===== (B) SIRV без оптимізації =============================================
λ, βv, μv, δv = 1e-4, 0.6, 1e-4, 0.2
θ, η, ζ, ξ, τ = 0.2, 0.05, 0.01, 0.1, 0.01
S2 = np.zeros_like(t); V2 = np.zeros_like(t)
I2 = np.zeros_like(t); R2 = np.zeros_like(t)
S2[0], I2[0], R2[0] = 950, 50, 0

def dS2(s, v, i): return (1-θ)*λ*N - βv*s*i/N + ζ*v - η*s - μv*s
def dV2(s, v, i): return θ*λ*N + η*s - ξ*βv*v*i/N - (ζ+μv)*v
def dI2(s, v, i): return βv*s*i/N + ξ*βv*v*i/N - (τ+δv+μv)*i
def dR2(i, r):     return δv*i - μv*r

for k in range(N_steps):
    k1S, k1V, k1I, k1R = dS2(S2[k],V2[k],I2[k]), dV2(S2[k],V2[k],I2[k]), dI2(S2[k],V2[k],I2[k]), dR2(I2[k],R2[k])
    k2S = dS2(S2[k]+0.5*dt*k1S, V2[k]+0.5*dt*k1V, I2[k]+0.5*dt*k1I)
    k2V = dV2(S2[k]+0.5*dt*k1S, V2[k]+0.5*dt*k1V, I2[k]+0.5*dt*k1I)
    k2I = dI2(S2[k]+0.5*dt*k1S, V2[k]+0.5*dt*k1V, I2[k]+0.5*dt*k1I)
    k2R = dR2(I2[k]+0.5*dt*k1I, R2[k]+0.5*dt*k1R)
    k3S = dS2(S2[k]+0.5*dt*k2S, V2[k]+0.5*dt*k2V, I2[k]+0.5*dt*k2I)
    k3V = dV2(S2[k]+0.5*dt*k2S, V2[k]+0.5*dt*k2V, I2[k]+0.5*dt*k2I)
    k3I = dI2(S2[k]+0.5*dt*k2S, V2[k]+0.5*dt*k2V, I2[k]+0.5*dt*k2I)
    k3R = dR2(I2[k]+0.5*dt*k2I, R2[k]+0.5*dt*k2R)
    k4S = dS2(S2[k]+dt*k3S, V2[k]+dt*k3V, I2[k]+dt*k3I)
    k4V = dV2(S2[k]+dt*k3S, V2[k]+dt*k3V, I2[k]+dt*k3I)
    k4I = dI2(S2[k]+dt*k3S, V2[k]+dt*k3V, I2[k]+dt*k3I)
    k4R = dR2(I2[k]+dt*k3I, R2[k]+dt*k3R)
    S2[k+1] = S2[k] + (dt/6)*(k1S+2*k2S+2*k3S+k4S)
    V2[k+1] = V2[k] + (dt/6)*(k1V+2*k2V+2*k3V+k4V)
    I2[k+1] = I2[k] + (dt/6)*(k1I+2*k2I+2*k3I+k4I)
    R2[k+1] = R2[k] + (dt/6)*(k1R+2*k2R+2*k3R+k4R)

# ===== (C) базова SIR ========================================================
S3 = np.zeros_like(t); I3 = np.zeros_like(t); R3 = np.zeros_like(t)
S3[0], I3[0], R3[0] = 950, 50, 0

def dS3(s, i): return lam*N - beta*s*i/N - mu*s
def dI3(s, i): return beta*s*i/N - delta*i - mu*i
def dR3(i, r): return delta*i - mu*r

for k in range(N_steps):
    k1S, k1I, k1R = dS3(S3[k],I3[k]), dI3(S3[k],I3[k]), dR3(I3[k],R3[k])
    k2S = dS3(S3[k]+0.5*dt*k1S, I3[k]+0.5*dt*k1I)
    k2I = dI3(S3[k]+0.5*dt*k1S, I3[k]+0.5*dt*k1I)
    k2R = dR3(I3[k]+0.5*dt*k1I, R3[k]+0.5*dt*k1R)
    k3S = dS3(S3[k]+0.5*dt*k2S, I3[k]+0.5*dt*k2I)
    k3I = dI3(S3[k]+0.5*dt*k2S, I3[k]+0.5*dt*k2I)
    k3R = dR3(I3[k]+0.5*dt*k2I, R3[k]+0.5*dt*k2R)
    k4S = dS3(S3[k]+dt*k3S, I3[k]+dt*k3I)
    k4I = dI3(S3[k]+dt*k3S, I3[k]+dt*k3I)
    k4R = dR3(I3[k]+dt*k3I, R3[k]+dt*k3R)
    S3[k+1] = S3[k] + (dt/6)*(k1S+2*k2S+2*k3S+k4S)
    I3[k+1] = I3[k] + (dt/6)*(k1I+2*k2I+2*k3I+k4I)
    R3[k+1] = R3[k] + (dt/6)*(k1R+2*k2R+2*k3R+k4R)

# ===== (D) збереження графіків ==============================================
# -- (D1) u(t) сходинками -----------------------------------------------------
plt.figure(figsize=(6, 2.5))
plt.step(t, u_step, where="mid", lw=2, color="purple")
plt.xlabel("Дні"); plt.ylabel("u (частка щеплених/день)")
plt.title("Оптимальна інтенсивність вакцинації $u(t)$")
plt.ylim(-0.01, u_max*1.05); plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/optimal_vaccination_rate.png", dpi=300)
plt.close()

# -- допоміжна функція для окремих графіків -----------------------------------
def save_compare(y_opt, y_sirv, y_sir, name):
    plt.figure(figsize=(5.2, 3.2))
    plt.plot(t, y_opt,   label=f"{name} (opt)",  lw=2)
    plt.plot(t, y_sirv,  label=f"{name} (SIRV)", lw=1.8)
    plt.plot(t, y_sir,  '--', label=f"{name} (SIR)",  lw=1.6)
    plt.title(f"{name}(t)"); plt.xlabel("Дні"); plt.ylabel("Осіб")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"figures/compare_{name}.png", dpi=300)
    plt.close()

save_compare(S_opt, S2, S3, "S")
save_compare(I_opt, I2, I3, "I")
save_compare(R_opt, R2, R3, "R")

print("✅  Збережено: optimal_vaccination_rate.png, compare_S.png, compare_I.png, compare_R.png у папці ./figures/")

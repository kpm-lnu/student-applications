import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

λ = 10
δS = 0.1
δI = 0.2
β = 0.005
π = 5
γ = 0.5

S0 = 80
I0 = 10
V0 = 5
T = 20
t_eval = np.linspace(0, T, 500)


def model_no_control(t, y):
    S, I, V = y
    dSdt = λ - δS * S - β * V * S
    dIdt = β * V * S - δI * I
    dVdt = π * I - γ * V
    return [dSdt, dIdt, dVdt]


sol_no_control = solve_ivp(model_no_control, [0, T], [S0, I0, V0], t_eval=t_eval)

plt.figure(figsize=(10, 5))
plt.plot(sol_no_control.t, sol_no_control.y[0], label='S(t) - здорові')
plt.plot(sol_no_control.t, sol_no_control.y[1], label='I(t) - інфіковані')
plt.plot(sol_no_control.t, sol_no_control.y[2], label='V(t) - віріони')
plt.title('Без лікування')
plt.xlabel('Час')
plt.ylabel('Популяція')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


A = 1
B = 0.1
C = 0.2

def calculate_optimal_control(t, S, I, ψ1, ψ2, current_umax1, current_umax2):
    u1 = min(current_umax1, max(0, ψ1 * S / B))
    u2 = min(current_umax2, max(0, ψ2 * I / C))
    return u1, u2

def run_forward_backward_sweep(umax1_val, umax2_val):
    N = len(t_eval)
    h = T / (N - 1)
    x = np.zeros((3, N))  
    ψ = np.zeros((3, N))  
    u1 = np.zeros(N)
    u2 = np.zeros(N)


    x[:, 0] = [S0, I0, V0]
    ψ[:, -1] = [0, 0, 0]


    u1.fill(0.5 * umax1_val) 
    u2.fill(0.5 * umax2_val)

    for _ in range(100): 

        for i in range(N - 1):
            S, I, V = x[:, i]

            current_u1_est, current_u2_est = calculate_optimal_control(t_eval[i], S, I, ψ[0, i], ψ[1, i], umax1_val, umax2_val)
            u1[i] = current_u1_est
            u2[i] = current_u2_est


            dS = λ - δS * S - β * V * S - u1[i] * S
            dI = β * V * S - (δI + u2[i]) * I
            dV = π * I - γ * V
            

            x[0, i + 1] = S + h * dS
            x[1, i + 1] = I + h * dI
            x[2, i + 1] = V + h * dV


        for i in reversed(range(N - 1)):
            S_next, I_next, V_next = x[:, i+1] 
            ψ1_next, ψ2_next, ψ3_next = ψ[:, i + 1] 

          
            current_u1 = u1[i] 
            current_u2 = u2[i]

         
            dψ1 = ψ1_next * (δS + β * V_next + current_u1) - ψ2_next * β * V_next
            dψ2 = -A + ψ2_next * (δI + current_u2) - ψ3_next * π
            dψ3 = ψ1_next * β * S_next - ψ2_next * β * S_next + ψ3_next * γ
            
            ψ[0, i] = ψ1_next - h * dψ1
            ψ[1, i] = ψ2_next - h * dψ2
            ψ[2, i] = ψ3_next - h * dψ3
    

    S_last, I_last, V_last = x[:, -1]
    ψ1_last, ψ2_last, ψ3_last = ψ[:, -1]
    u1[-1], u2[-1] = calculate_optimal_control(t_eval[-1], S_last, I_last, ψ1_last, ψ2_last, umax1_val, umax2_val)

    return x, u1, u2


def calculate_J(I_values, u1_values, u2_values, h):
    # Функціонал J = integral(AI(t) + 0.5*B*u1(t)^2 + 0.5*C*u2(t)^2) dt
    integrand = A * I_values + 0.5 * B * (u1_values ** 2) + 0.5 * C * (u2_values ** 2)
    return np.trapz(integrand, dx=h) 


u_1_s1 = 1
u_2_s1 = 1
x_s1, u1_s1, u2_s1 = run_forward_backward_sweep(u_1_s1, u_2_s1)

plt.figure(figsize=(10, 5))
plt.plot(t_eval, x_s1[0], label='S(t) - здорові')
plt.plot(t_eval, x_s1[1], label='I(t) - інфіковані')
plt.plot(t_eval, x_s1[2], label='V(t) - віріони')
plt.title(f'З оптимальним керуванням (u_1={u_1_s1}, u_2={u_2_s1})')
plt.xlabel('Час')
plt.ylabel('Популяція')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

u_1_s2 = 0
u_2_s2 = 1
x_s2, u1_s2, u2_s2 = run_forward_backward_sweep(u_1_s2, u_2_s2)

plt.figure(figsize=(10, 5))
plt.plot(t_eval, x_s2[0], label='S(t) - здорові')
plt.plot(t_eval, x_s2[1], label='I(t) - інфіковані')
plt.plot(t_eval, x_s2[2], label='V(t) - віріони')
plt.title(f'З оптимальним керуванням (u_1={u_1_s2}, u_2={u_2_s2})')
plt.xlabel('Час')
plt.ylabel('Популяція')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

u_1_s3 = 1
u_2_s3 = 0
x_s3, u1_s3, u2_s3 = run_forward_backward_sweep(u_1_s3, u_2_s3)

plt.figure(figsize=(10, 5))
plt.plot(t_eval, x_s3[0], label='S(t) - здорові')
plt.plot(t_eval, x_s3[1], label='I(t) - інфіковані')
plt.plot(t_eval, x_s3[2], label='V(t) - віріони')
plt.title(f'З оптимальним керуванням (u_1={u_1_s3}, u_2={u_2_s3})')
plt.xlabel('Час')
plt.ylabel('Популяція')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

h_step = T / (len(t_eval) - 1)
J_s1 = calculate_J(x_s1[1], u1_s1, u2_s1, h_step)
J_s2 = calculate_J(x_s2[1], u1_s2, u2_s2, h_step)
J_s3 = calculate_J(x_s3[1], u1_s3, u2_s3, h_step)

data = [
    ["u_1=1, u_2=1", f"{J_s1:.2f}"],
    ["u_1=0, u_2=1", f"{J_s2:.2f}"],
    ["u_1=1, u_2=0", f"{J_s3:.2f}"]
]

columns = ["Випадок", "J_value"]

fig, ax = plt.subplots(figsize=(6, 2)) 

ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=data,
                     colLabels=columns,
                     loc='center',
                     cellLoc='center',
                     colColours=["#f2f2f2", "#f2f2f2"]) 

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2) 

ax.set_title("Значення функціоналу цілі J для різних Випадків", fontsize=14, pad=20)

plt.show()
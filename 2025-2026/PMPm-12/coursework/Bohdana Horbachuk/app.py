import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. АРХІТЕКТУРА НЕЙРОННОЇ МЕРЕЖІ
# ==========================================
class IntegralEquationNN(nn.Module):
    def __init__(self, hidden_layers=3, neurons=50):
        super(IntegralEquationNN, self).__init__()
        layers = []
        layers.append(nn.Linear(1, neurons))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

# ==========================================
# 2. МАТЕМАТИЧНА МОДЕЛЬ
# ==========================================
def gamma2(t, example_id):
    if example_id == 1:
        return torch.tensor([math.cos(t), math.sin(t) + 2.0])
    elif example_id == 2:
        r = math.sqrt(math.cos(t)**2 + 0.25 * math.sin(t)**2)
        return torch.tensor([r * math.cos(t), r * math.sin(t) + 1.5])
    elif example_id == 3:
        return torch.tensor([2.0 * math.cos(t), 0.5 * math.sin(t) + 1.2])

def f1(x):
    return torch.exp(-x[0]**2)

def f2(x):
    return 1.0

def Phi(x, y):
    return torch.log(1.0 / torch.norm(x - y))

def G(x, y):
    y2 = torch.tensor([y[0], -y[1]])
    return Phi(x, y) - Phi(x, y2)

def H1(ti, tj):
    return -0.5

def H(ti, tj, example_id):
    return G(gamma2(ti, example_id), gamma2(tj, example_id))

def H2(ti, tj, example_id):
    val_i = gamma2(ti, example_id)
    val_j = gamma2(tj, example_id)
    
    if abs(ti - tj) < 1e-8:
        dt = 1e-5
        x_prime = (gamma2(ti + dt, example_id) - gamma2(ti - dt, example_id)) / (2 * dt)
        val_i_star = torch.tensor([val_i[0], -val_i[1]])
        val1 = Phi(val_i, val_i_star)
        return 0.5 * torch.log(1.0 / (math.exp(1) * torch.norm(x_prime)**2)) + val1
    else:
        return H(ti, tj, example_id) - H1(ti, tj) * math.log(4.0 / math.exp(1) * math.sin((ti - tj) / 2.0)**2)
    
def R_weight(ti, tj, M):
    val = 0.0
    for m in range(1, M):
        val += math.cos(m * (ti - tj)) / m
    return -1.0 / (2 * M) * (1.0 + 2.0 * val + math.cos(M * (ti - tj)) / M)

def L_kernel(x, tau):
    y = torch.tensor([tau, 0.0])
    return -2.0 * x[1] / (torch.norm(x - y)**2)

def vectorB(ti, example_id, M_inf, h_inf):
    x = gamma2(ti, example_id)
    res = 0.0
    for j in range(-M_inf, M_inf + 1):
        t_j = j * h_inf
        res += f1(torch.tensor([t_j, 0.0])) * L_kernel(x, t_j)
    return f2(x) + (res * h_inf) / (2 * math.pi)

def generate_collocation_data(N_points, example_id):
    M = N_points // 2 
    t_vals = torch.tensor([i * math.pi / M for i in range(N_points)], dtype=torch.float32)
    
    A = torch.zeros((N_points, N_points), dtype=torch.float32)
    F = torch.zeros((N_points, 1), dtype=torch.float32)
    
    M_inf = 1000
    h_inf = 1.0 / math.sqrt(M_inf)
    
    for i in range(N_points):
        ti = t_vals[i].item()
        F[i, 0] = vectorB(ti, example_id, M_inf, h_inf)
        for j in range(N_points):
            tj = t_vals[j].item()
            A[i, j] = H1(ti, tj) * R_weight(ti, tj, M) + 1.0 / (2 * M) * H2(ti, tj, example_id)
    return t_vals.view(-1, 1), A, F

# ==========================================
# 3. ІНТЕРФЕЙС ТА НАВЧАННЯ МОДЕЛІ
# ==========================================
st.set_page_config(page_title="PINN: Інтегральні рівняння Лапласа", layout="wide")
st.title("Розв'язання інтегрального рівняння Лапласа за допомогою штучної нейронної мережі")

st.sidebar.header("Параметри математичної моделі")
example_choice = st.sidebar.selectbox(
    "Оберіть конфігурацію межі (Γ2):",
    ("Приклад 1: Зміщене коло", "Приклад 2: Область 'Гантеля'", "Приклад 3: Еліптична область")
)
example_id = int(example_choice.split(":")[0][-1])

st.sidebar.markdown("---")
st.sidebar.header("Параметри дискретизації")
N_report = st.sidebar.select_slider("Кількість вузлів (N):", options=[4, 8, 16, 32], value=32)
N_points = 2 * N_report 

st.sidebar.markdown("---")
st.sidebar.header("Гіперпараметри нейромережі")
hidden_layers = st.sidebar.slider("Кількість прихованих шарів", 1, 6, 3)
neurons = st.sidebar.slider("Кількість нейронів у шарі", 10, 100, 50)
epochs = st.sidebar.slider("Кількість епох навчання", 500, 10000, 3000, step=500)
lr = st.sidebar.number_input("Швидкість навчання (Learning Rate)", value=0.005, format="%.4f")

if st.sidebar.button("🚀 Ініціалізувати навчання"):
    st.write(f"### Процес оптимізації для конфігурації: {example_choice.split(':')[1]}...")
    
    t_data, A_matrix, F_star = generate_collocation_data(N_points, example_id)
    
    model = IntegralEquationNN(hidden_layers=hidden_layers, neurons=neurons)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty() 
    
    loss_history = []
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        mu_pred = model(t_data) 
        left_side = torch.matmul(A_matrix, mu_pred)
        loss = loss_fn(left_side, F_star)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % (max(1, epochs // 20)) == 0:
            progress_bar.progress(epoch / epochs)
            status_text.text(f"Епоха: {epoch}/{epochs} | MSE Loss: {loss.item():.6e}")
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(loss_history, color='red')
            ax.set_yscale('log')
            ax.set_title("Динаміка функції втрат (Loss) в процесі навчання")
            ax.set_xlabel("Епохи")
            ax.set_ylabel("MSE Loss")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            chart_placeholder.pyplot(fig)
            plt.close(fig)
            
    st.success("✅ Процес навчання успішно завершено.")
    
    st.write("### Апроксимована густина потенціалу $\mu(t)$")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t_data.detach().numpy(), mu_pred.detach().numpy(), label="Апроксимація $\mu(t)$", color='blue', linewidth=2)
    ax2.set_xlabel("$t \in [0, 2\pi]$")
    ax2.set_ylabel("Значення $\mu(t)$")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    st.session_state['is_trained'] = True
    st.session_state['mu_pred'] = mu_pred
    st.session_state['t_data'] = t_data
    st.session_state['example_id'] = example_id
    st.session_state['N_points'] = N_points
    st.session_state['A_matrix'] = A_matrix
    st.session_state['F_star'] = F_star


# ==========================================
# 4. ВЕРИФІКАЦІЯ РЕЗУЛЬТАТІВ (INFERENCE)
# ==========================================
if st.session_state.get('is_trained', False):
    st.markdown("---")
    st.header("🎯 Верифікація наближеного розв'язку в заданій точці області")
    
    col1, col2 = st.columns(2)
    with col1:
        test_x1 = st.number_input("Введіть координату $x_1$:", value=4.0)
    with col2:
        test_x2 = st.number_input("Введіть координату $x_2$:", value=2.0)
        
    if st.button("Обчислити значення потенціалу"):
        mu_pred_saved = st.session_state['mu_pred']
        t_data_saved = st.session_state['t_data']
        ex_id_saved = st.session_state['example_id']
        N_pts_saved = st.session_state['N_points']
        A_matrix_saved = st.session_state['A_matrix']
        F_star_saved = st.session_state['F_star']
        
        x_test = torch.tensor([test_x1, test_x2], dtype=torch.float32)
        M = N_pts_saved // 2
        
        u_exact = None
        if ex_id_saved == 1 and test_x1 == 4.0 and test_x2 == 1.0:
            u_exact = 0.1398515749
        elif ex_id_saved == 1 and test_x1 == 4.0 and test_x2 == 2.0:
            u_exact = 0.2450365158
        elif ex_id_saved == 1 and test_x1 == 0.0 and test_x2 == 4.0:
            u_exact = 0.7077864615
        elif ex_id_saved == 2 and test_x1 == 2.0 and test_x2 == 2.0:
            u_exact = 0.5263440473
        elif ex_id_saved == 2 and test_x1 == 4.0 and test_x2 == 2.0:
            u_exact = 0.2108171043

        # Розв'язок класичним чисельним методом (СЛАР)
        mu_classic = torch.linalg.solve(A_matrix_saved, F_star_saved).squeeze()
        
        u_classic_gamma2 = 0.0
        u_approx_gamma2 = 0.0 
        
        for j in range(N_pts_saved):
            tj = t_data_saved[j].item()
            G_val = G(x_test, gamma2(tj, ex_id_saved)).item()
            u_classic_gamma2 += mu_classic[j].item() * G_val
            u_approx_gamma2 += mu_pred_saved[j].item() * G_val
            
        u_classic_gamma2 = u_classic_gamma2 / (2 * M)
        u_approx_gamma2 = u_approx_gamma2 / (2 * M)
        
        # Sinc-квадратура для нескінченної межі
        M_inf = 1000
        h_inf = 1.0 / math.sqrt(M_inf)
        u_gamma1 = 0.0
        
        for j in range(-M_inf, M_inf + 1):
            t_j = j * h_inf
            u_gamma1 += f1(torch.tensor([t_j, 0.0])).item() * L_kernel(x_test, t_j).item()
        u_gamma1 = (u_gamma1 * h_inf) / (2 * math.pi)
        
        u_classic_final = u_classic_gamma2 - u_gamma1
        u_approx_final = u_approx_gamma2 - u_gamma1
        
        st.subheader("Результати обчислень:")
        
        st.write(f"🔹 **Класичний метод (Nyström/СЛАР):** `{u_classic_final:.10f}`")
        st.write(f"🔸 **Запропонований метод (Нейромережа):** `{u_approx_final:.10f}`")
        
        diff = abs(u_classic_final - u_approx_final)
        st.write(f"**Абсолютна розбіжність між методами:** `{diff:.8e}`")
        
        if u_exact is not None:
            st.markdown("---")
            st.write(f"📚 **Аналітичний еталон:** `{u_exact:.10f}`")
            error_classic = abs(u_exact - u_classic_final)
            error_nn = abs(u_exact - u_approx_final)
            st.write(f"Похибка класичного методу відносно еталона: `{error_classic:.8e}`")
            st.write(f"Похибка нейромережевого алгоритму відносно еталона: `{error_nn:.8e}`")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# DeepPINN Model Definition
# =============================================================================
class DeepPINN(nn.Module):
    def __init__(self, layers):
        super(DeepPINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        a = x
        for i in range(len(self.layers)-1):
            a = self.activation(self.layers[i](a))
        output = self.layers[-1](a)
        return output

# =============================================================================
# PDE Residual for Burgers' Equation
# =============================================================================
def compute_pde_residual(model, x, t, nu):
    # Form the input (x,t) with shape (N,2)
    inputs = torch.cat([x, t], dim=1)
    u = model(inputs)
    # Compute first derivatives using autograd
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        retain_graph=True, create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        retain_graph=True, create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # Burgers' equation residual: u_t + u*u_x - nu*u_xx = 0
    f = u_t + u * u_x - nu * u_xx
    return f

# =============================================================================
# Data Generation Functions
# =============================================================================
def generate_training_data(N_collocation=10000, N_boundary=200, N_initial=200):
    # Use a structured grid for collocation points
    sqrt_Nc = int(np.sqrt(N_collocation))
    x_coll = torch.linspace(-1, 1, sqrt_Nc).unsqueeze(1)
    t_coll = torch.linspace(0, 1, sqrt_Nc).unsqueeze(1)
    X_coll, T_coll = torch.meshgrid(x_coll.squeeze(), t_coll.squeeze(), indexing='ij')
    X_coll = X_coll.reshape(-1, 1)
    T_coll = T_coll.reshape(-1, 1)
    
    # Initial condition: u(x,0) = -sin(pi*x)
    x_initial = torch.linspace(-1, 1, N_initial).unsqueeze(1)
    t_initial = torch.zeros_like(x_initial)
    u_initial = -torch.sin(np.pi * x_initial)
    
    # Boundary conditions: u(-1,t)=0 and u(1,t)=0
    t_boundary = torch.linspace(0,1, N_boundary).unsqueeze(1)
    x_boundary_left = -torch.ones_like(t_boundary)
    x_boundary_right = torch.ones_like(t_boundary)
    u_boundary_left = torch.zeros_like(t_boundary)
    u_boundary_right = torch.zeros_like(t_boundary)
    
    return (X_coll.to(device), T_coll.to(device),
            x_initial.to(device), t_initial.to(device), u_initial.to(device),
            x_boundary_left.to(device), t_boundary.to(device), u_boundary_left.to(device),
            x_boundary_right.to(device), t_boundary.to(device), u_boundary_right.to(device))

def generate_test_data(N_collocation=2000, N_boundary=200, N_initial=200):
    # For testing, sample collocation points uniformly at random.
    x_coll = torch.FloatTensor(N_collocation, 1).uniform_(-1, 1)
    t_coll = torch.FloatTensor(N_collocation, 1).uniform_(0, 1)
    
    # Same procedure as training for initial and boundary conditions.
    x_initial = torch.linspace(-1, 1, N_initial).unsqueeze(1)
    t_initial = torch.zeros_like(x_initial)
    u_initial = -torch.sin(np.pi * x_initial)
    
    t_boundary = torch.linspace(0, 1, N_boundary).unsqueeze(1)
    x_boundary_left = -torch.ones_like(t_boundary)
    x_boundary_right = torch.ones_like(t_boundary)
    u_boundary_left = torch.zeros_like(t_boundary)
    u_boundary_right = torch.zeros_like(t_boundary)
    
    return (x_coll.to(device), t_coll.to(device),
            x_initial.to(device), t_initial.to(device), u_initial.to(device),
            x_boundary_left.to(device), t_boundary.to(device), u_boundary_left.to(device),
            x_boundary_right.to(device), t_boundary.to(device), u_boundary_right.to(device))

# =============================================================================
# Accuracy Metric Functions
# =============================================================================
def compute_regression_accuracy(u_pred, u_true, tol=0.05):
    """
    Computes the percentage of points where the relative error is below tol.
    For u_true nonzero: relative error = |u_pred - u_true|/(|u_true|+1e-6)
    """
    relative_error = torch.abs(u_pred - u_true) / (torch.abs(u_true) + 1e-6)
    acc = torch.mean((relative_error < tol).float()) * 100.0
    return acc.item()

def compute_boundary_accuracy(u_pred, u_true, tol=0.05):
    """
    For boundary points with true value 0,
    we use an absolute tolerance.
    """
    acc = torch.mean((torch.abs(u_pred - u_true) < tol).float()) * 100.0
    return acc.item()

def evaluate_test(model, test_data, nu, mse_loss):
    (x_coll, t_coll, 
     x_initial, t_initial, u_initial,
     x_boundary_left, t_boundary_left, u_boundary_left,
     x_boundary_right, t_boundary_right, u_boundary_right) = test_data
     
    # Ensure x_coll and t_coll are leaf tensors with gradients enabled
    x_coll = x_coll.clone().detach().requires_grad_(True)
    t_coll = t_coll.clone().detach().requires_grad_(True)
    
    # Collocation loss (PDE residual)
    f_pred = compute_pde_residual(model, x_coll, t_coll, nu)
    loss_coll = mse_loss(f_pred, torch.zeros_like(f_pred))
    
    # Initial condition loss
    inputs_initial = torch.cat([x_initial, t_initial], dim=1)
    u_pred_initial = model(inputs_initial)
    loss_initial = mse_loss(u_pred_initial, u_initial)
    
    # Boundary losses
    inputs_boundary_left = torch.cat([x_boundary_left, t_boundary_left], dim=1)
    u_pred_boundary_left = model(inputs_boundary_left)
    loss_boundary_left = mse_loss(u_pred_boundary_left, u_boundary_left)
    
    inputs_boundary_right = torch.cat([x_boundary_right, t_boundary_right], dim=1)
    u_pred_boundary_right = model(inputs_boundary_right)
    loss_boundary_right = mse_loss(u_pred_boundary_right, u_boundary_right)
    
    total_loss = loss_coll + loss_initial + loss_boundary_left + loss_boundary_right
    
    # Compute accuracy metrics on initial condition and boundaries
    acc_initial = compute_regression_accuracy(u_pred_initial, u_initial, tol=0.05)
    acc_boundary_left = compute_boundary_accuracy(u_pred_boundary_left, u_boundary_left, tol=0.05)
    acc_boundary_right = compute_boundary_accuracy(u_pred_boundary_right, u_boundary_right, tol=0.05)
    
    overall_accuracy = (acc_initial + acc_boundary_left + acc_boundary_right) / 3
    return total_loss.item(), overall_accuracy

# =============================================================================
# Hyperparameters, Model, Loss, and Optimizer Setup
# =============================================================================
layers = [2, 50, 50, 50, 50, 1]  # Input (x,t), several hidden layers, output u
nu = 0.01 / np.pi  # viscosity parameter
model = DeepPINN(layers).to(device)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 5000
evaluation_interval = 100  # evaluate on test data every 100 epochs

# Generate datasets
train_data = generate_training_data()
test_data = generate_test_data()

# =============================================================================
# Training Loop with Metric Storage
# =============================================================================
train_loss_history = []
test_loss_history = []
test_accuracy_history = []
print("Starting training...")

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Unpack training data
    (x_coll, t_coll,
     x_initial, t_initial, u_initial,
     x_boundary_left, t_boundary_left, u_boundary_left,
     x_boundary_right, t_boundary_right, u_boundary_right) = train_data
    
    # Ensure collocation points allow gradients
    x_coll.requires_grad = True
    t_coll.requires_grad = True
    
    # PDE residual loss
    f_pred = compute_pde_residual(model, x_coll, t_coll, nu)
    loss_f = mse_loss(f_pred, torch.zeros_like(f_pred))
    
    # Initial condition loss
    inputs_initial = torch.cat([x_initial, t_initial], dim=1)
    u_pred_initial = model(inputs_initial)
    loss_initial = mse_loss(u_pred_initial, u_initial)
    
    # Boundary condition losses
    inputs_boundary_left = torch.cat([x_boundary_left, t_boundary_left], dim=1)
    u_pred_boundary_left = model(inputs_boundary_left)
    loss_boundary_left = mse_loss(u_pred_boundary_left, u_boundary_left)
    
    inputs_boundary_right = torch.cat([x_boundary_right, t_boundary_right], dim=1)
    u_pred_boundary_right = model(inputs_boundary_right)
    loss_boundary_right = mse_loss(u_pred_boundary_right, u_boundary_right)
    
    # Total training loss
    loss = loss_f + loss_initial + loss_boundary_left + loss_boundary_right
    loss.backward()
    optimizer.step()
    
    train_loss_history.append(loss.item())
    
    # Every few epochs, evaluate on test data
    if epoch % evaluation_interval == 0:
        model.eval()
        test_loss, test_accuracy = evaluate_test(model, test_data, nu, mse_loss)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
        print(f"Epoch {epoch}: Train Loss: {loss.item():.5e}, Test Loss: {test_loss:.5e}, Test Accuracy: {test_accuracy:.2f}%")

print("Training finished.")

# =============================================================================
# Evaluation and Visualization of the Predicted Solution
# =============================================================================
model.eval()

# Create a fine grid over the domain for visualization
nx, nt = 256, 100  # resolution for x and t
x = np.linspace(-1, 1, nx)
t = np.linspace(0, 1, nt)
X, T = np.meshgrid(x, t)  # X and T have shape (nt, nx)

# Prepare inputs for the model: each row is an (x,t) pair.
X_star = np.hstack((X.reshape(-1, 1), T.reshape(-1, 1)))
X_star_tensor = torch.tensor(X_star, dtype=torch.float32).to(device)

with torch.no_grad():
    u_pred_grid = model(X_star_tensor).cpu().numpy()

U = u_pred_grid.reshape(T.shape)

plt.figure(figsize=(8,6))
contour = plt.contourf(X, T, U, levels=100, cmap='jet')
plt.xlabel('x')
plt.ylabel('t')
plt.title("Predicted Solution u(x,t) of Burgers' Equation via PINN")
plt.colorbar(contour, label='u(x,t)')
plt.show()

# =============================================================================
# Plotting Training and Test Losses and Test Accuracy History
# =============================================================================
# Plotting the loss history
plt.figure(figsize=(8,6))
plt.plot(train_loss_history, 'ro-', label="Train Loss", alpha=0.7)
eval_epochs = np.arange(0, num_epochs, evaluation_interval)
plt.plot(eval_epochs, test_loss_history, 'go-', label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss History")
plt.legend()
plt.grid(True)
plt.show()

# Plotting the test accuracy history
plt.figure(figsize=(8,6))
plt.plot(eval_epochs, test_accuracy_history, 'go-', label="Test Accuracy (%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy History")
plt.legend()
plt.grid(True)
plt.show()

# Save the model state for future use
torch.save(model.state_dict(), 'pinn_burgers.pth')

# Save the model architecture
with open('pinn_burgers_architecture.txt', 'w') as f:
    f.write(str(layers))

# Save the training and test data
train_data_np = [tensor.detach().cpu().numpy() for tensor in train_data]
test_data_np = [tensor.detach().cpu().numpy() for tensor in test_data]
np.savez('training_data.npz', *train_data_np)
np.savez('test_data.npz', *test_data_np)

# Save the training and test loss history
np.savez('loss_history.npz', train_loss=train_loss_history, test_loss=test_loss_history, test_accuracy=test_accuracy_history)

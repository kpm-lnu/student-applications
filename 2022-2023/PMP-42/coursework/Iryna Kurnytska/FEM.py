import numpy as np
import matplotlib.pyplot as plt
from void_mesh import *
from function import *
from post_process import *
import matplotlib.colors as mcolors
from datetime import datetime
from uniform_mesh import *
import abc


class AbstractMesh(abc.ABC):
    @abc.abstractmethod
    def generate_mesh(self):
        pass


class AbstractPostProcess(abc.ABC):
    @abc.abstractmethod
    def post_process(self):
        pass


class BoatMesh(AbstractMesh):
    def __init__(self, d1, d2, p, m, R, element_type):
        self.d1 = d1
        self.d2 = d2
        self.p = p
        self.m = m
        self.R = R
        self.element_type = element_type

    def generate_mesh(self):
        if self.element_type == "D2QU4N":
            return void_mesh(self.d1, self.d2, self.p, self.m, self.R, self.element_type)
        elif self.element_type == "D2TR3N":
            return void_mesh(self.d1, self.d2, self.p, self.m, self.element_type)
        else:
            raise ValueError("Invalid element type specified.")


class BoatPostProcess(AbstractPostProcess):
    def post_process(self, NL, EL, ENL):
        # Виконати обробку результатів

        (stress_xx, stress_xy, stress_yx, stress_yy, strain_xx, strain_xy,
         strain_yx, strain_yy, disp_x, disp_y, X, Y) = post_process(NL, EL, ENL)

        stress_xxNormalized = (stress_xx - stress_xx.min()) / (stress_xx.max() - stress_xx.min())
        disp_xNormalized = (disp_x - disp_x.min()) / (disp_x.max() - disp_x.min())

        # Повернути оброблені результати
        return stress_xxNormalized, disp_xNormalized, X, Y

class AbstractVisualizer:
    def __init__(self, title, subplot):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(subplot)
        self.ax.set_title(title)

    def plot(self, x, y, c):
        cmap = truncate_colormap(plt.get_cmap('jet'), c.min(), c.max())
        t = self.ax.tripcolor(x, y, c, cmap=cmap, shading='gouraud')
        p = self.ax.plot(x, y, 'k-', linewidth=0.5)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap






startTime = datetime.now()

d1 = 1
d2 = 2
p = 4
m = 3
R = 0.3
element_type = "D2QU4N"
# element_type="D2TR3N"
defV = 0.3

# Generate mesh
mesh = BoatMesh(d1, d2, p, m, R, element_type)
NL, EL = mesh.generate_mesh()

NoN = np.size(NL, 0)
NoE = np.size(EL, 0)

plt.figure(3)

count = 1  # Annotate nodes numbers
for i in range(0, NoN):
    plt.annotate(count, xy=(NL[i, 0], NL[i, 1]))
    count += 1

if element_type == "D2QU4N":
    count2 = 1  # annotate element numbers
    for j in range(0, NoE):
        plt.annotate(count2, xy=(
            (NL[EL[j, 0] - 1, 0] + NL[EL[j, 1] - 1, 0] + NL[EL[j, 2] - 1, 0] + NL[EL[j, 3] - 1, 0]) / 4,
            (NL[EL[j, 0] - 1, 1] + NL[EL[j, 1] - 1, 1] + NL[EL[j, 2] - 1, 1] + NL[EL[j, 3] - 1, 1]) / 4),
                     c="blue")
        count2 += 1

    # plot lines
    x0, y0 = NL[EL[:, 0] - 1, 0], NL[EL[:, 0] - 1, 1]
    x1, y1 = NL[EL[:, 1] - 1, 0], NL[EL[:, 1] - 1, 1]
    x2, y2 = NL[EL[:, 2] - 1, 0], NL[EL[:, 2] - 1, 1]
    x3, y3 = NL[EL[:, 3] - 1, 0], NL[EL[:, 3] - 1, 1]

    plt.plot(np.array([x0, x1]), np.array([y0, y1]), "black", linewidth=3)
    plt.plot(np.array([x1, x2]), np.array([y1, y2]), "black", linewidth=3)
    plt.plot(np.array([x2, x3]), np.array([y2, y3]), "black", linewidth=3)
    plt.plot(np.array([x3, x0]), np.array([y3, y0]), "black", linewidth=3)

    plt.show()

elif element_type == "D2TR3N":
    count2 = 1  # annotate element numbers
    for j in range(0, NoE):
        plt.annotate(count2, xy=(
            (NL[EL[j, 0] - 1, 0] + NL[EL[j, 1] - 1, 0] + NL[EL[j, 2] - 1, 0]) / 3,
            (NL[EL[j, 0] - 1, 1] + NL[EL[j, 1] - 1, 1] + NL[EL[j, 2] - 1, 1]) / 3),
                     c="blue")

        count2 += 1

    # plot lines
    x0, y0 = NL[EL[:, 0] - 1, 0], NL[EL[:, 0] - 1, 1]
    x1, y1 = NL[EL[:, 1] - 1, 0], NL[EL[:, 1] - 1, 1]
    x2, y2 = NL[EL[:, 2] - 1, 0], NL[EL[:, 2] - 1, 1]

    plt.plot(np.array([x0, x1]), np.array([y0, y1]), "black", linewidth=3)
    plt.plot(np.array([x1, x2]), np.array([y1, y2]), "black", linewidth=3)
    plt.plot(np.array([x2, x0]), np.array([y2, y0]), "black", linewidth=3)

    plt.show()


# BC_flag='extension'
BC_flag = 'expansion'
# BC_flag='shear'

# Assign BCs
(ENL, DOFs, DOCs) = assign_BCs(NL, BC_flag, defV)

# Assemble stiffness matrix, forces, and displacements
K = assemble_stiffness(ENL, EL, NL)
Fp = assemble_forces(ENL, NL)
Up = assemble_displacements(ENL, NL)

K_reduced = K[0:DOFs, 0:DOFs]
K_UP = K[0:DOFs, DOFs:DOCs + DOFs]
K_PU = K[DOFs:DOCs + DOFs, 0:DOFs]
K_PP = K[DOFs:DOCs + DOFs, DOFs:DOCs + DOFs]

F = Fp - (K_UP @ Up)
Uu = np.linalg.solve(K_reduced, F)

Fu = (K_PU @ Uu) + (K_PP @ Up)
ENL = update_nodes(ENL, Uu, Fu, NL)

# Post-process
post_processor = BoatPostProcess()
stress_xxNormalized, disp_xNormalized , X,Y= post_processor.post_process(NL, EL, ENL)

fig_1 = AbstractVisualizer('Stress_xx', 111)
fig_2 = AbstractVisualizer('Displacement X', 111)

for i in range(np.size(EL, 0)):
    x = X[:, i]
    y = Y[:, i]
    c = stress_xxNormalized[:, i]
    fig_1.plot(x, y, c)

for i in range(np.size(EL, 0)):
    x = X[:, i]
    y = Y[:, i]
    c = disp_xNormalized[:, i]
    fig_2.plot(x, y, c)

print(datetime.now() - startTime)

plt.show()

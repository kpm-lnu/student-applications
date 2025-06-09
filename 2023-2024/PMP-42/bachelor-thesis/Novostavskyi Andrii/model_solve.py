import tkinter as tk

from matplotlib.pyplot import xticks
from matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.font_manager import FontProperties
from numpy.core.multiarray import ndarray
import numpy as np
from pylab import figure, plot, xlabel, grid, legend, title, ylabel
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


fields = [
  ('a_1', 0.7, 'Коефіцієнт'),
  ('B_1', 0.3, 'Коефіцієнт'),
  ('delta_1', 1, 'Коефіцієнт'),
  ('sigma_1', 0.5, 'Коефіцієнт'),
  ('a_3', 0.98, 'Коефіцієнт'),
  ('B_2', 0.7, 'Коефіцієнт'),
  ('y_2', 0.8, 'Коефіцієнт'),
  ('sigma_2', 0.4, 'Коефіцієнт'),
  ('s', 0.4, 'Коефіцієнт'),
  ('p', 0.2, 'Коефіцієнт'),
  ('w', 0.3, 'Коефіцієнт'),
  ('y_3', 0.3, 'Коефіцієнт'),
  ('u', 0.29, 'Коефіцієнт'),
  ('sigma_3', 0.5, 'Коефіцієнт'),
  ('v', 1, 'Коефіцієнт'),
  ('pi', 0.5, 'Коефіцієнт'),
  ('theta', 0.98, 'Коефіцієнт'),
  ('H_0', 1, 'Початковий стан'),
  ('T_0', 0.00001, 'Початковий стан'),
  ('I_0', 1.379310345, 'Початковий стан'),
  ('E_0', 0.5, 'Початковий стан'),
  ('t_max', 25, 'Час'),
]


coefficient_names = [
  "a_1",
  "B_1",
  "delta_1",
  "sigma_1",
  "a_3",
  "B_2",
  "y_2",
  "sigma_2",
  "s",
  "p",
  "w",
  "y_3",
  "u",
  "sigma_3",
  "v",
  "pi",
  "theta",
]


initial_state_variable_names = [
   "H_0",
   "T_0",
   "I_0",
   "E_0",
]


def ode_fun(t, state_var, *coefficients):
    H, T, I, E = state_var
    (a_1, B_1, delta_1, sigma_1, a_3, B_2, y_2, sigma_2, s, p, w, y_3, u, sigma_3, v, pi, theta) = coefficients
    dHdt = H * (a_1 - B_1 * H - delta_1 * T) - sigma_1 * H * E
    dTdt = T * (a_3 - B_2 * T) - y_2 * I * T + sigma_2 * H * E
    dIdt = s + (p * I * T) / (w + T) - y_3 * I * T - u * I - (sigma_3 * I * E) / (v + E)
    dEdt = pi - theta * E
    return [dHdt, dTdt, dIdt, dEdt]


def draw_result(t, state_variables):
    plt.figure(figsize=(12, 6))

    # Use rcParams to set the font properties globally
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})

    plt.xlabel('$t$')
    plt.ylabel('$Клітини$')

    plt.grid(True)

    lw = 3  # Set linewidth to a thicker value
    plt.plot(t, state_variables[0], 'b', linewidth=lw)
    plt.plot(t, state_variables[1], 'r', linewidth=lw)
    plt.plot(t, state_variables[2], 'g', linewidth=lw)

    plt.xticks(range(0, 26, 5))

    legend_properties = FontProperties(size=16, weight='bold')
    plt.legend((r'$H(t)$', r'$T(t)$', r'$I(t)$'), loc='upper right', bbox_to_anchor=(1.1, 1), prop=legend_properties)

    # Setting font weight to bold using fontdict for title
    plt.title('Динаміка популяцій клітин', fontdict={'weight': 'bold'})

    pylab.show()


entries = {}


def solve_ode():
    coefficients = fetch_from_entries(coefficient_names)
    state_variables0 = fetch_from_entries(initial_state_variable_names)
    stoptime, = fetch_from_entries(['t_max'])

    sol = solve_ivp(ode_fun, [0, stoptime], state_variables0, args=coefficients, method='RK45', dense_output=True)
    t = np.linspace(0, stoptime, 1000)
    state_variables = sol.sol(t)
    draw_result(t, state_variables)

def fetch_from_entries(coefficients):
  values = []
  for c in coefficients:
      values.append(float(entries[c].get()))
  return values


def add_hint(field, hint):
  def show_hint():
      tk.messagebox.showinfo(field, hint)
  return show_hint


def makeform(root, fields):
    entries = {}
    for col, (field, value, hint) in enumerate(fields):
        row = tk.Frame(root)
        tk.Label(row, width=20, text=field, anchor='w').grid(row=0, column=col*2)
        ent = tk.Entry(row)
        ent.insert(0, value)
        ent.grid(row=0, column=col*2+1)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        entries[field] = ent
    return entries

# ...

if __name__ == '__main__':
    root = tk.Tk()
    entries = makeform(root, fields)
    b1 = tk.Button(root, text='Show', command=solve_ode)
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root, text='Quit', command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()




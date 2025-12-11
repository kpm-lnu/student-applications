"""
OPF_course_project_simple.py

Два методи розв'язання задачі OPF для мережі IEEE 14-bus:
 - AC-OPF (повна нелінійна модель)
 - DC-OPF (лінійна спрощена модель)

Реалізація на основі pandapower:
 - pp.runopp   -> AC Optimal Power Flow
 - pp.rundcopp -> DC Optimal Power Flow

Після розрахунків будується кілька графіків для порівняння результатів.
"""

import copy
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt


# ---------------------------
# 1. Завантаження тестової мережі
# ---------------------------

def load_test_network():
    """
    Завантажує стандартну тестову мережу IEEE 14-bus з pandapower.
    """
    net = pn.case14()

    # На всякий випадок задаємо межі напруг, якщо вони не задані
    if "max_vm_pu" not in net.bus:
        net.bus["max_vm_pu"] = 1.05
    if "min_vm_pu" not in net.bus:
        net.bus["min_vm_pu"] = 0.95

    return net


# ---------------------------
# 2. AC-OPF
# ---------------------------

def run_ac_opf(net):
    """
    Виконує AC-OPF для мережі net за допомогою pandapower.runopp.
    Повертає:
      - копію мережі net_ac з результатами
      - таблиці результатів по вузлах, лініях, генераторах та опорному джерелу.
    """
    net_ac = copy.deepcopy(net)

    # AC-OPF (метод внутрішньої точки всередині pandapower)
    pp.runopp(net_ac, calculate_voltage_angles=True)

    res_bus = net_ac.res_bus.copy()
    res_line = net_ac.res_line.copy()
    res_gen = net_ac.res_gen.copy()
    res_ext = net_ac.res_ext_grid.copy()

    return net_ac, res_bus, res_line, res_gen, res_ext


# ---------------------------
# 3. DC-OPF
# ---------------------------

def run_dc_opf(net):
    """
    Виконує DC-OPF для мережі net за допомогою pandapower.rundcopp.
    Повертає:
      - копію мережі net_dc з результатами
      - таблиці результатів по вузлах, лініях, генераторах та опорному джерелу.
    """
    net_dc = copy.deepcopy(net)

    # DC-OPF (внутрішня лінійна модель у pandapower)
    pp.rundcopp(net_dc, calculate_voltage_angles=True)

    res_bus = net_dc.res_bus.copy()
    res_line = net_dc.res_line.copy()
    res_gen = net_dc.res_gen.copy()
    res_ext = net_dc.res_ext_grid.copy()

    return net_dc, res_bus, res_line, res_gen, res_ext


# ---------------------------
# 4. Візуалізація результатів
# ---------------------------

def plot_results(res_bus_ac, res_line_ac, res_gen_ac, res_ext_ac,
                 res_bus_dc, res_line_dc, res_gen_dc, res_ext_dc,
                 net_ac, net_dc, res_bus_before):
    """
    Будує базові графіки для порівняння AC-OPF і DC-OPF:
      1) Генерація по вузлах (AC vs DC).
      2) Напруги по вузлах (AC).
      3) Потоки по лініях (AC vs DC).
    """

    # ---------- 1. Генерація по вузлах AC vs DC ----------

    # AC: збираємо Pg по bus із res_gen_ac + res_ext_ac
    ac_gen_bus = {}

    # Генератори
    for idx, row in res_gen_ac.iterrows():
        bus = int(net_ac.gen.loc[idx, "bus"])
        ac_gen_bus[bus] = ac_gen_bus.get(bus, 0.0) + row["p_mw"]

    # Опорне джерело (ext_grid)
    for idx, row in res_ext_ac.iterrows():
        bus = int(net_ac.ext_grid.loc[idx, "bus"])
        ac_gen_bus[bus] = ac_gen_bus.get(bus, 0.0) + row["p_mw"]

    # DC: те саме
    dc_gen_bus = {}
    for idx, row in res_gen_dc.iterrows():
        bus = int(net_dc.gen.loc[idx, "bus"])
        dc_gen_bus[bus] = dc_gen_bus.get(bus, 0.0) + row["p_mw"]

    for idx, row in res_ext_dc.iterrows():
        bus = int(net_dc.ext_grid.loc[idx, "bus"])
        dc_gen_bus[bus] = dc_gen_bus.get(bus, 0.0) + row["p_mw"]

    all_buses = sorted(set(ac_gen_bus.keys()).union(dc_gen_bus.keys()))
    ac_vals = [ac_gen_bus.get(b, 0.0) for b in all_buses]
    dc_vals = [dc_gen_bus.get(b, 0.0) for b in all_buses]


    plt.figure()
    plt.plot(all_buses, ac_vals, marker="o", label="AC-OPF")
    plt.plot(all_buses, dc_vals, marker="x", linestyle="--", label="DC-OPF")
    plt.xlabel("Номер вузла (bus)")
    plt.ylabel("Активна генерація, MW")
    plt.title("Порівняння генерації AC-OPF vs DC-OPF")
    plt.grid(True)
    plt.legend()

    # ---------- 2. Напруги по вузлах (AC) ----------

    # ---------- Напруги ДО та ПІСЛЯ AC-OPF ----------


    plt.figure()
    plt.plot(res_bus_before.index, res_bus_before["vm_pu"], marker="o", label="До OPF")
    plt.plot(res_bus_ac.index, res_bus_ac["vm_pu"], marker="x", label="Після AC-OPF")
    plt.xlabel("Номер вузла (bus)")
    plt.ylabel("Напруга, p.u.")
    plt.title("Порівняння напруг у вузлах: до OPF і після AC-OPF")
    plt.grid(True)
    plt.legend()

    # ---------- 3. Потоки по лініях AC vs DC ----------

    # AC-потоки: беремо топологію з net_ac.line та результати з res_line_ac
    ac_lines = net_ac.line[["from_bus", "to_bus"]].copy()
    ac_lines["p_from_mw"] = res_line_ac["p_from_mw"].values
    ac_lines["key"] = list(zip(ac_lines["from_bus"], ac_lines["to_bus"]))
    ac_flows = ac_lines.set_index("key")["p_from_mw"]

    # DC-потоки: аналогічно з net_dc.line і res_line_dc
    dc_lines = net_dc.line[["from_bus", "to_bus"]].copy()
    dc_lines["p_from_mw"] = res_line_dc["p_from_mw"].values
    dc_lines["key"] = list(zip(dc_lines["from_bus"], dc_lines["to_bus"]))
    dc_flows = dc_lines.set_index("key")["p_from_mw"]

    keys = sorted(set(ac_flows.index).union(dc_flows.index))
    ac_f = [ac_flows.get(k, 0.0) for k in keys]
    dc_f = [dc_flows.get(k, 0.0) for k in keys]
    line_labels = [f"{i}-{j}" for (i, j) in keys]
    x = np.arange(len(keys))

    plt.figure()
    plt.plot(x, ac_f, marker="o", label="AC-OPF")
    plt.plot(x, dc_f, marker="x", linestyle="--", label="DC-OPF")
    plt.xticks(x, line_labels, rotation=90)
    plt.xlabel("Лінія (from-to)")
    plt.ylabel("Потік активної потужності, MW")
    plt.title("Потоки по лініях: AC-OPF vs DC-OPF")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()



# ---------------------------
# 5. main()
# ---------------------------

def main():
    net = load_test_network()
    print(net)

    # ---------------------------------------
    # 1. Напруги ДО оптимізації
    # ---------------------------------------
    print("\nВиконую звичайний Load Flow до OPF…")
    pp.runpp(net)
    res_bus_before = net.res_bus.copy()

    # ---- AC-OPF ----
    net_ac, res_bus_ac, res_line_ac, res_gen_ac, res_ext_ac = run_ac_opf(net)
    print("AC-OPF завершено.")
    print("Сумарна активна генерація (AC):",
          res_gen_ac["p_mw"].sum() + res_ext_ac["p_mw"].sum(), "MW")

    # ---- DC-OPF ----
    net_dc, res_bus_dc, res_line_dc, res_gen_dc, res_ext_dc = run_dc_opf(net)
    print("DC-OPF завершено.")
    print("Сумарна активна генерація (DC):",
          res_gen_dc["p_mw"].sum() + res_ext_dc["p_mw"].sum(), "MW")

    # ---- Візуалізація ----
    plot_results(res_bus_ac, res_line_ac, res_gen_ac, res_ext_ac,
                 res_bus_dc, res_line_dc, res_gen_dc, res_ext_dc,
                 net_ac, net_dc, res_bus_before)


if __name__ == "__main__":
    main()

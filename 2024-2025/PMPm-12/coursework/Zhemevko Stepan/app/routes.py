from flask import Blueprint, render_template, request, redirect, url_for
from app.model import create_default_graph, load_graph_from_csv
from app.optimizer import optimize_energy_network, analyze_solution, greedy_energy_distribution, \
    dijkstra_energy_routing
from app.visualizer import draw_graph_interactive


main = Blueprint('main', __name__)
G_current = None  # глобальний граф для візуалізації, оптимізації тощо

# === Головна сторінка ===
@main.route('/')
def index():
    global G_current
    G_current = create_default_graph()
    return render_template('index.html')

# === Завантаження CSV ===
@main.route('/upload', methods=['POST'])
def upload():
    global G_current
    file = request.files.get('file')
    method = request.form.get('method', 'lp')
    if not file:
        return redirect(url_for('main.index'))

    G_current = load_graph_from_csv(file)
    return redirect(url_for('main.results', method=method))

@main.route('/results')
def results():
    global G_current
    if not G_current:
        return redirect(url_for('main.index'))

    method = request.args.get('method', 'lp')
    import copy
    G_copy = copy.deepcopy(G_current)

    if method == 'greedy':
        result = greedy_energy_distribution(G_copy)
        analysis = analyze_solution(G_copy, result['flows'])
    elif method == 'dijkstra':
        result = dijkstra_energy_routing(G_copy)
        analysis = analyze_solution(G_copy, result['flows'])
    else:
        result = optimize_energy_network(G_copy)
        analysis = analyze_solution(G_copy, result['flows'])

    edge_data = [
        {
            "from": u,
            "to": v,
            "capacity": d['capacity'],
            "loss": d['loss'],
            "flow": result['flows'].get(f"{u}->{v}", 0)
        }
        for u, v, d in G_copy.edges(data=True)
    ]

    node_data = [
        {
            "node": n,
            "type": d.get('type'),
            "supply": d.get('supply', 0),
            "demand": d.get('demand', 0)
        }
        for n, d in G_copy.nodes(data=True)
    ]

    method_name_map = {
        "lp": "Лінійне програмування (LP)",
        "greedy": "Жадібний алгоритм",
        "heuristic": "Евристичний метод",
        "dijkstra": "Маршрутизація Дейкстри"
    }
    selected_method_name = method_name_map.get(method, "Невідомий метод")

    return render_template(
        'results.html',
        result=result,
        analysis=analysis,
        edges=edge_data,
        nodes=node_data,
        selected_method=selected_method_name
    )

# === Візуалізація графа ===
@main.route('/visualize')
def visualize():
    global G_current
    if not G_current:
        return redirect(url_for('main.index'))
    graph_html = draw_graph_interactive(G_current)
    return render_template('visualize.html', graph_html=graph_html)

# === Сторінка з метриками (опціональна, якщо є metrics.html) ===
@main.route('/metrics')
def metrics():
    global G_current
    if not G_current:
        return redirect(url_for('main.index'))

    result = optimize_energy_network(G_current)
    analysis = analyze_solution(G_current, result['flows'])

    return render_template('metrics.html', analysis=analysis, result=result)

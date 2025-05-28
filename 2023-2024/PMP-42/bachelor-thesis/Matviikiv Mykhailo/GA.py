import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.use('TkAgg')
counter = 0


class Config:
    def __init__(self):
        self.pop_size = 60
        self.p_c = 0.6
        self.p_m = 0.2
        self.N_el = 0.1
        self.N_t = 2
        self.max_it = 500
        self.max_it_nc = 50


def get_data(path):
    date = []
    infected = []
    death = []
    recovered = []
    with open(path, 'r') as f:
        for x in f.readlines():
            temp = x.split()
            if len(temp) != 5:
                continue
            date.append(temp[0])
            infected.append(int(temp[2]))
            death.append(int(temp[3]))
            recovered.append(int(temp[4]))
    return date, infected, death, recovered


date, i, d, r = get_data("2020.txt")
data = [date, i, r]
config = Config()
N = 41.1e6  # Total population of Ukraine
days = len(data[0])  # Number of days for modeling
E0 = 0  # Initial exposed individuals
I0 = i[0]  # Initial number of infected individuals
R0 = r[0]  # Initial number of recovered or deceased individuals

population = []
repro_limits = [1.5, 4.0]
tinf_limits = [4, 30]
tinc_limits = [2, 30]


def SEIR_model_brovchenko(N, E0, I0, R0, repro, tinf, tinc, days, draw=False):
    global counter
    counter += 1

    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0] = N - I0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    for i in range(1, days):
        dSdt = -repro / tinf * (1 / N) * S[i - 1] * I[i - 1]
        dEdt = (repro / tinf) * (1 / N) * S[i - 1] * I[i - 1] - (E[i - 1] / tinc)
        dIdt = (E[i - 1] / tinc) - (I[i - 1] / tinf)
        dRdt = I[i - 1] / tinf
        S[i] = S[i - 1] + dSdt
        E[i] = E[i - 1] + dEdt
        I[i] = I[i - 1] + dIdt
        R[i] = R[i - 1] + dRdt
    if draw:
        date, i, d, r = get_data("2020.txt")
        # plt.plot(S, label="Здорові")
        # plt.plot(E, label="Інкубаційні")
        plt.plot(I, label="Інфіковані(result)")
        # plt.plot(R, label=f"Одужані(result)")
        plt.plot(i, label="Інфіковані(DB)")
        # plt.plot(r, label=f"Одужані(DB)")
        plt.xlabel("Дні")
        plt.ylabel("Кількість людей")
        plt.title("SEIR модель для COVID-19 в Україні")
        plt.legend()
        plt.show()
    return S, E, I, R


def run():
    start = time.time()
    generate_initial_population()
    infos = [get_population_info()]
    for i in range(2, config.max_it):
        generate_new_population()
        infos.append(get_population_info())
        if check_no_changes(infos, i):
            break
    best_info = infos[-1]
    print(
        f'solution after {i} iterations: {", ".join(map(str, best_info["best"]["chromosome"]))}, f = {best_info["best_fitness"]}')
    end = time.time()
    print("time: ", end - start)
    print(counter)
    # plot_result(infos)
    S, E, I, R = SEIR_model_brovchenko(N, E0, I0, R0, best_info["best"]["chromosome"][0],
                                       best_info["best"]["chromosome"][1], best_info["best"]["chromosome"][2], days,
                                       draw=True)


def check_no_changes(infos, current_iteration):
    _, mnldx = min((info['best_fitness'], idx) for idx, info in enumerate(infos))
    return (current_iteration - mnldx) > config.max_it_nc


def plot_result(infos):
    iterations = list(range(1, len(infos) + 1))
    best_fitnesses = [info['best_fitness'] for info in infos]
    plt.plot(iterations, best_fitnesses, linewidth=3, label='best')
    best_idx = np.argmin(best_fitnesses)
    plt.scatter(iterations[best_idx], best_fitnesses[best_idx], color='blue', s=70,
                label=f'best individual\n{iterations[best_idx], best_fitnesses[best_idx]}')
    plt.legend(fontsize=14)
    plt.xlabel('Iterations', fontsize=13)
    plt.ylabel('Fitness', fontsize=13)
    plt.show()


def generate_initial_population():
    global population
    repro_values = np.linspace(repro_limits[0], repro_limits[1], num=config.pop_size)
    tinf_values = np.linspace(tinf_limits[0], tinf_limits[1], num=config.pop_size)
    tinc_values = np.linspace(tinc_limits[0], tinc_limits[1], num=config.pop_size)
    np.random.shuffle(repro_values)
    np.random.shuffle(tinf_values)
    np.random.shuffle(tinc_values)
    param_list = list(zip(repro_values, tinf_values, tinc_values))
    population = [{'chromosome': param, 'fitness': compute_functional(param[0], param[1], param[2])} for param in
                  param_list]


def get_population_info():
    global population
    best_idx = np.argmin([individual['fitness'] for individual in population])
    best_individual = population[best_idx]
    return {'best': best_individual,
            'best_fitness': best_individual['fitness']}


def compute_cumprobs():
    global config
    global population

    fitnesses = np.array([individual['fitness'] for individual in population])
    transformed_fitnesses = np.max(fitnesses) - fitnesses + 1
    probs = transformed_fitnesses / np.sum(transformed_fitnesses)
    config.cumprobs = np.cumsum(probs)


def compute_functional(repro, tinf, tinc):
    global N, days, E0, I0, R0

    S, E, I, R = SEIR_model_brovchenko(N, E0, I0, R0, repro, tinf, tinc, days)

    summ = 0
    for i in range(days):
        summ += (((I[i] - data[1][i]) ** 2) + ((R[i] - data[2][i]) ** 2))

    return summ / days


def select_individual():
    global config
    global population

    r = np.random.rand()
    idx = np.where(r <= config.cumprobs)[0][0]
    individual = population[idx]
    return individual


def do_crossover(p1, p2):
    global config
    if np.random.rand() > config.p_c:
        return p1, p2
    c1 = []
    c2 = []
    a = [np.random.rand() for _ in range(len(p1))]
    for i in range(len(p1)):
        c1.append(a[i] * p1[i] + (1 - a[i]) * p2[i])
        c2.append((1 - a[i]) * p1[i] + a[i] * p2[i])
    # як зробити так шоб після crossover не вийшов за межі
    return tuple(c1), tuple(c2)


def mutate(chromosome):
    global config
    if np.random.rand() > config.p_m:
        return chromosome
    k = np.random.randint(0, 3)
    mut_chrom = [x for x in chromosome]
    if k == 0:
        m_val = min(mut_chrom[k] - repro_limits[0], repro_limits[1] - mut_chrom[k])
        mut_chrom[k] += np.random.uniform(-m_val, m_val)
        np.clip(mut_chrom[2], repro_limits[0], repro_limits[1])
    if k == 1:
        m_val = min(mut_chrom[k] - tinf_limits[0], tinf_limits[1] - mut_chrom[k])
        mut_chrom[k] += np.random.uniform(-m_val, m_val)
        np.clip(mut_chrom[2], tinf_limits[0], tinf_limits[1])
    if k == 2:
        m_val = min(mut_chrom[k] - tinc_limits[0], tinc_limits[1] - mut_chrom[k])
        mut_chrom[k] += np.random.uniform(-m_val, m_val)
        np.clip(mut_chrom[2], tinc_limits[0], tinc_limits[1])
    return tuple(mut_chrom)


def generate_new_population():
    global population
    compute_cumprobs()
    new_population = get_elit_individuals()
    for i in range(len(new_population) // 2 + 1, int(config.pop_size / 2) + 1):  # moment
        parent1 = select_individual()
        parent2 = select_individual()
        child1_chromosome, child2_chromosome = do_crossover(parent1['chromosome'], parent2['chromosome'])
        child1_chromosome = mutate(child1_chromosome)
        child2_chromosome = mutate(child2_chromosome)
        child1_fitness = compute_functional(child1_chromosome[0], child1_chromosome[1], child1_chromosome[2])
        child2_fitness = compute_functional(child2_chromosome[0], child2_chromosome[1], child2_chromosome[2])
        child1 = {'chromosome': child1_chromosome, 'fitness': child1_fitness}
        child2 = {'chromosome': child2_chromosome, 'fitness': child2_fitness}
        new_population.append(child1)
        new_population.append(child2)
    population = new_population[:config.pop_size]


def get_elit_individuals():
    global population
    sorted_population = sorted(population, key=lambda x: x['fitness'])
    return sorted_population[:int(config.N_el * config.pop_size)]


run()

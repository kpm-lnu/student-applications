import csv
import random
import matplotlib.pyplot as plt


class Chromosome:
    def __init__(self, num_matches):
        self.genes = [self.weighted_random_choice() for _ in range(num_matches)]

    @staticmethod
    def weighted_random_choice():
        choices = [0, 1, 2, 4, 5, 10, 15, 20]
        weights = [10, 10, 10, 5, 3, 2, 1, 1]
        return (random.choices(choices, weights)[0], random.choices(choices, weights)[0])

    def mutate(self, mutation_rate):
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] = self.mutate_gene(self.genes[i])

    @staticmethod
    def mutate_gene(gene):
        choices = [0, 1, 2, 4, 5, 10, 15, 20]
        weights = [10, 10, 10, 5, 3, 2, 1, 1]
        mutated_gene = (max(0, min(20, gene[0] + random.choices(choices, weights)[0])),
                        max(0, min(20, gene[1] + random.choices(choices, weights)[0])))
        return mutated_gene

    def to_prediction(self):
        return self.genes


def load_data_from_csv(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def make_prediction(best_individual, future_match, data):
    home_team = future_match['home_team']
    away_team = future_match['away_team']
    home_matches = [match for match in data if match['home_team'] == home_team]
    away_matches = [match for match in data if match['away_team'] == away_team]
    home_points = evaluate_team_performance(best_individual, home_matches)
    away_points = evaluate_team_performance(best_individual, away_matches)
    print(f"Home team ({home_team}) points: {home_points}, Away team ({away_team}) points: {away_points}")

    if home_points > away_points:
        return f"Winner: {home_team}"
    elif home_points < away_points:
        return f"Winner: {away_team}"
    else:
        return f"Draw"


def evaluate_team_performance(best_individual, matches):
    total_points = 0
    for match in matches:
        home_score = int(match['home_score'])
        away_score = int(match['away_score'])
        prediction = best_individual.genes[matches.index(match)]

        predicted_home_score, predicted_away_score = prediction
        if home_score == predicted_home_score and away_score == predicted_away_score:
            total_points += 3
        elif (home_score > away_score and predicted_home_score > predicted_away_score) or \
                (home_score < away_score and predicted_home_score < predicted_away_score) or \
                (home_score == away_score and predicted_home_score == predicted_away_score):
            total_points += 1
    return total_points


def selection(population, data):
    population.sort(key=lambda x: evaluate_team_performance(x, data), reverse=True)
    return population[:len(population) // 3]


def tournament_selection(population, data, k=2):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, k)
        best = max(tournament, key=lambda x: evaluate_team_performance(x, data))
        selected.append(best)
    return selected


def crossover(parent1, parent2):
    if random.random() < 0.8:
        crossover_point = random.randint(0, len(parent1.genes) - 1)
        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]
    else:
        child1_genes = parent1.genes
        child2_genes = parent2.genes

    child1 = Chromosome(len(child1_genes))
    child2 = Chromosome(len(child2_genes))
    child1.genes = child1_genes
    child2.genes = child2_genes
    return child1, child2


def adaptive_mutation_rate(generation, max_generations):
    return max(0.01, 0.1 - 0.09 * (generation / max_generations))


def genetic_algorithm(data, population_size, mutation_rate, num_generations):
    population = [Chromosome(len(data)) for _ in range(population_size)]
    best_individual = None
    max_fitness_values = []
    avg_fitness_values = []
    for generation in range(num_generations):
        elite_size = population_size // 3  # 33% елітних індивідів
        elite = selection(population, data)[:elite_size]
        selected = tournament_selection(population, data)
        new_population = elite.copy()
        while len(new_population) < population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            if parent1 != parent2:
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([child1, child2])
        for individual in new_population:
            individual.mutate(adaptive_mutation_rate(generation, num_generations))
        population = new_population

        fitness_scores = [evaluate_team_performance(ind, data) for ind in population]
        max_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        max_fitness_values.append(max_fitness)
        avg_fitness_values.append(avg_fitness)

        best_individual = max(population, key=lambda x: evaluate_team_performance(x, data))
        print(
            f"Generation {generation + 1}: Best Individual - {best_individual.genes}, Points: {evaluate_team_performance(best_individual, data)}")

    return best_individual, max_fitness_values, avg_fitness_values


def plot_fitness(max_fitness_values, avg_fitness_values):
    generations = list(range(1, len(max_fitness_values) + 1))
    plt.plot(generations, max_fitness_values, label="Макс. пристосованість", color='red')
    plt.plot(generations, avg_fitness_values, label="Серед. пристосованість", color='blue')
    plt.xlabel("Покоління")
    plt.ylabel("Пристосованість")
    plt.title("Максимальна та середня пристосованість за поколіннями")
    plt.legend()
    plt.show()


def main():
    file_path = 'results.csv'
    data = load_data_from_csv(file_path)
    population_size = 200
    mutation_rate = 0.1
    num_generations = 120
    best_individual, max_fitness_values, avg_fitness_values = genetic_algorithm(data, population_size, mutation_rate,
                                                                                num_generations)

    total_points = evaluate_team_performance(best_individual, data)
    print("Total points for the best prediction: ", total_points)

    future_match1 = {'home_team': 'Scotland', 'away_team': 'England'}
    future_match2 = {'home_team': 'England', 'away_team': 'Scotland'}

    prediction1 = make_prediction(best_individual, future_match1, data)
    prediction2 = make_prediction(best_individual, future_match2, data)

    print("Prediction for future match 1:", prediction1)
    print("Prediction for future match 2:", prediction2)

    plot_fitness(max_fitness_values, avg_fitness_values)


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt


def generate_random_network(n_users, p_follow):
    # network = np.random.choice([0, 1], size=(n_users, n_users), p=[1-p_follow, p_follow])
    # np.fill_diagonal(network, 0)
    network = np.zeros((n_users, n_users))
    random_numbers = np.random.rand(n_users, n_users)
    for i in range(n_users):
        for j in range(i + 1, n_users):
            mutual_friends = np.sum(network[i] * network[j])

            adjusted_p_follow = p_follow + (1 - p_follow) * (mutual_friends / n_users)

            if random_numbers[i, j] < adjusted_p_follow:
                network[i, j] = 1
                network[j, i] = 1
    return network


def infect_followers(network, infected_users, cured_users, p_infection):
    new_infected_users = set()
    for user, is_infected in enumerate(infected_users):
        if is_infected == 1:
            followers = np.nonzero(network[:, user])[0]
            for follower in followers:
                if np.random.rand() < p_infection and infected_users[follower] == 0 and cured_users[follower] == 0:
                    new_infected_users.add(follower)
    return new_infected_users


def cure_followers(network, infected_users, cured_users, p_cure_infected, p_cure_uninfected):
    new_cured_users = set()
    for user, is_infected in enumerate(infected_users):
        if cured_users[user] == 1:
            continue
        if is_infected:
            if np.random.rand() < p_cure_infected:
                cured_users[user] = 1
                new_cured_users.add(user)
        # else:
        #     if np.random.rand() < p_cure_uninfected:
        #         cured_users[user] = 1
        #         infected_users[user] = 0
    return new_cured_users


def generate():
    np.random.seed(47)

    n_users = 1000
    p_follow = 0.033

    p_infection = 10 / n_users  # Початкова ймовірність зараження підписника в разі зараження користувача
    p_cure_infected = 0.01
    p_cure_uninfected = 0.01

    decay_rate = 0.0001  # Швидкість зменшення ймовірності зараження з часом
    # noise_infection_rate = 0.01  # Ймовірність випадкової шумової інфекції
    n_iter = 200
    n_infected = int(n_users / 100) * 3

    network = generate_random_network(n_users, p_follow)
    # for row in network:
    #     print(np.sum(row))

    infected_indices = np.random.choice(n_users, size=n_infected, replace=False)
    infected_users = np.zeros(n_users, dtype=int)
    infected_users[infected_indices] = 1
    cured_users = np.zeros(n_users)

    n_infected_on_iter = np.zeros(n_iter + 1)
    n_infected_on_iter[0] = np.sum(infected_users)
    n_cured_on_iter = np.zeros(n_iter + 1)

    total_infected_users = np.zeros(n_iter + 1)
    total_infected_users[0] = np.sum(infected_users)
    total_cured_users = np.zeros(n_iter + 1)

    for step in range(1, n_iter + 1):
        new_infected_users = infect_followers(network, infected_users, cured_users, p_infection)
        total_infected_users[step] = total_infected_users[step - 1] + len(new_infected_users)
        n_infected_on_iter[step] = len(new_infected_users)
        infected_users[list(new_infected_users)] = 1
        p_infection -= decay_rate
        new_cured_users = cure_followers(network, infected_users, cured_users, p_cure_infected, p_cure_uninfected)
        total_cured_users[step] = total_cured_users[step - 1] + len(new_cured_users)
        # if np.random.rand() < noise_infection_rate:
        #     random_infected = np.random.randint(0, n_users-1)
        #     while infected_users[random_infected] != 0:
        #         random_infected = np.random.randint(0, n_users)
        #     n_infected_on_iter[random_infected] += 1
    # print("Кількість інфікованих на кожному кроці: ")
    # print(n_infected_on_iter)
    all_infected_on_iter = np.cumsum(n_infected_on_iter)
    # print("Загальна кількість інфікованих на кожному кроці: ")
    # print(all_infected_on_iter)
    # print("Загальна кількість вилікуваних на кожному кроці:")
    # print(total_cured_users)
    i = []
    s = []
    for infected, cured in zip(all_infected_on_iter, total_cured_users):
        i.append(infected - cured)
        s.append(n_users - infected)

    # x = list(range(0, 201))
    # plt.plot(x, s, label='S')
    # plt.plot(x, i, label='I')
    # plt.plot(x, total_cured_users, label='R')

    # plt.xlabel('Time')
    # plt.ylabel('Population')
    # plt.title('Generated data')
    
    # plt.legend()

    # plt.show()
    return np.array((s, i, total_cured_users))




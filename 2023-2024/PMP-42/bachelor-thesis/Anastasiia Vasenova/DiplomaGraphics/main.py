import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_data(allH, meanH):
    iterations = np.arange(len(allH))
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, allH, label='Максимальне значення пристосованості', color='red')
    plt.plot(iterations, meanH, label='Середнє значення пристосованості', color='blue')
    plt.xlabel('Ітерація')
    plt.ylabel('Похибка')
    plt.legend()
    plt.grid(True)
    plt.show()

def read_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        data = [float(line.replace(',', '.').strip()) for line in lines if line.strip()]
    return data

def main():
    allH = read_data("D:\\diploma_results\\bestH.txt")
    meanH = read_data("D:\\diploma_results\\meanH.txt")

    allH.pop(0)
    meanH.pop(0)

    plot_data(allH, meanH)

if __name__ == "__main__":
    main()

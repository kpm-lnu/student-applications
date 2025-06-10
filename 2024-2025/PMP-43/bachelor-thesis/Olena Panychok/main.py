import argparse
import os
from datetime import datetime
from examples import classic_example, multiple_interfaces_example, inclined_interface_example, convergence_study_classic, convergence_study, cfield_multiple_interfaces, cfield_inclined_interface

def make_results_dir(example_num):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"ex{example_num}_{now}"
    os.makedirs(dirname, exist_ok=True)
    return dirname

def main():
    parser = argparse.ArgumentParser(description="Seismic wave propagation examples")
    parser.add_argument('--example', type=int, default=1, help='Номер прикладу: 1 - класичний, 2 - множинні інтерфейси, 3 - похилий інтерфейс')
    args = parser.parse_args()
    ex = args.example
    results_dir = make_results_dir(ex)
    print(f"Всі результати будуть збережені у папці: {results_dir}")
    if ex == 1:
        os.chdir(results_dir)
        classic_example()
        convergence_study_classic()
    elif ex == 2:
        os.chdir(results_dir)
        multiple_interfaces_example()
        convergence_study("multiple_interfaces", cfield_multiple_interfaces)
    elif ex == 3:
        os.chdir(results_dir)
        inclined_interface_example()
        convergence_study("inclined_interface", cfield_inclined_interface)
    else:
        print("Невідомий номер прикладу. Оберіть 1, 2 або 3.")

if __name__ == "__main__":
    main()
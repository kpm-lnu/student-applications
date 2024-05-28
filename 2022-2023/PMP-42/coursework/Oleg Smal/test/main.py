import os
import shutil
import argparse


def main():
    start, stop = 1000, 10000
    tests = [start]
    logarithmic_increment = 0.1
    while tests[-1] < stop:
        tests.append(int(tests[-1] * (1 + logarithmic_increment)))
    tests[-1] = stop
    if not os.path.exists("./test/results"):
        os.mkdir("./test/results")
    source = "./buff"
    copy = ["cpu-profile.png", "memory-profile.png", "time-profile.txt"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="physics model")
    args = parser.parse_args()
    command = f"make profile_{args.model}"
    for test in tests:
        print(f"starting test #{test}...")
        os.system(f"./bin/init -overrideN={test}")
        os.system(command)
        destination = f"./test/results/{test}"
        if not os.path.exists(destination):
            os.mkdir(destination)
        for c in copy:
            shutil.move(f"{source}/{c}", f"{destination}/{c}")
        print(f"test #{test} finished.\n")


if __name__ == "__main__":
    main()
import random

if __name__ == "__main__":
    lines = open('randomFen.txt').read().splitlines()
    myline = random.choice(lines)
    print(myline)

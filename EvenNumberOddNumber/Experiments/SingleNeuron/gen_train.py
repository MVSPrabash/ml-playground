import argparse
import random

parser = argparse.ArgumentParser(description="Generate training data for Base10 Model")

parser.add_argument("p", type=int, help="Number of features (or digits)")
parser.add_argument("m", type=int, help="Number of training examples")

args = parser.parse_args()

p = args.p
m = args.m

l = 0
r = 10 ** p - 1

for _ in range(m):
    x = random.randint(0, r)
    print(x, int(x % 2 == 0))

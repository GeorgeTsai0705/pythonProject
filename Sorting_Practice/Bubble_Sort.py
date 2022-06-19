import random
import time

original = [random.randint(1, 300) for x in range(10)]
print(original)
start = time.time()

for j in range(len(original)):
    for i in range(len(original) - 1 - j):
        target = original[i]

        if original[i] > original[i + 1]:
            original[i] = original[i + 1]
            original[i + 1] = target

print(original)
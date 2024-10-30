import random
import time

original = [random.randint(1, 30000) for x in range(1000)]
print(original)
start = time.time()

gap = len(original) // 2
while gap>0:
    for i in range(gap, len(original)):
        temp = original[i]
        j = i
        while j >= gap and temp < original[j - gap]:
            original[j] = original[j-gap]
            j -= gap
        original[j] = temp
    gap = gap // 2

print("The time used to execute this is given below")
end = time.time()
print(end - start)
print(original)
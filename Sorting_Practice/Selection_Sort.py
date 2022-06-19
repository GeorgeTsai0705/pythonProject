import random
import time

original = [random.randint(1,30000) for x in range(2000)]

start = time.time()
sorted_set = []
for j in range(len(original)):
    Min_index = 0
    Min = original[Min_index]

    for i in range(len(original)):
        if original[i] < Min:
            Min_index = i
            Min = original[i]
    sorted_set.append(original.pop(Min_index))

print("The time used to execute this is given below")
end = time.time()
print(end - start)

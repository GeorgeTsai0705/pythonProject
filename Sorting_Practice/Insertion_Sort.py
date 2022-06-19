import random
import time

original = [random.randint(1,30000) for x in range(2000)]

start = time.time()
sorted_set = [original.pop(0)]

for j in range(len(original)):
    target = original.pop(0)
    inside = False

    for i in range(len(sorted_set)):
        if target < sorted_set[i]:
            sorted_set.insert(i, target)
            inside = True
            break
    if not inside:
        sorted_set.append(target)

print("The time used to execute this is given below")
end = time.time()
print(end - start)

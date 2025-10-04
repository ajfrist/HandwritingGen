import time

def use_power():
    a = 57 ** 45 + 57 ** 30 + 57 ** 15
    return a
start_time = time.time()
for i in range(10000):
    b = use_power()
end_time = time.time()
print(b)
print(f"Execution time power: {end_time - start_time} seconds")

def use_factor():
    a = 57**15 * (57**15 * (57**15 + 1)+ 1)
    return a
start_time = time.time()
for i in range(10000):
    b = use_factor()
end_time = time.time()
print(b)

print(f"Execution time factoring: {end_time - start_time} seconds")

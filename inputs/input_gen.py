import numpy as np

def gen_random(size):
    if size == "small":
        n = np.random.randint(76, 101)
        file = open("100.in", "w")
    elif size == "medium":
        n = np.random.randint(101, 151)
        file = open("150.in", "w")
    elif size == "large":
        n = np.random.randint(151, 201)
        file = open("200.in", "w")

    file.write(f"{n}\n")
    for i in range(1, n + 1):
        # Generate deadline
        t = np.random.randint(1, 1441)
        d = min(max(1, int(np.random.normal(30 , 20))), 60)
        p = np.round(np.random.uniform(0, 100), 3)
        file.write(f"{i} {t} {d} {p}\n")

    file.close()
    
gen_random("large")
    

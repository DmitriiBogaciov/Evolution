import random
import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools

creator.create("FitnessMax", base.Fitness, weights=(0.1,))

creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 10
toolbox = base.Toolbox()

toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n=10)
print(pop)
surface = 0.5

def evaluate(individual):
    fitness = sum(individual)
    fitness = fitness / IND_SIZE
    fitness += random.uniform(-1, 0.7)
    return fitness,


toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

NGEN = 10
CXPB = 0.9
MUTPB = 0.1

s = tools.Statistics(key=lambda ind: ind.fitness.values)
s.register("mean", np.mean)
s.register("max", np.max)

hof = tools.HallOfFame(1)  # pamatuje si 1 nejlepšího jedince za historii evoluce (i když zanikne)

pop = toolbox.population(n=10)

finalpop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=s, halloffame=hof)

mean, maximum = logbook.select("mean", "max")
best_pop = None
best_fitness = 0

print(hof)

fig, ax = plt.subplots()

ax.plot(range(0, (NGEN+1)*2, 2), mean, color='green', alpha=0.5, label="mean")

ax.axhline(y=surface, color='blue', linestyle='-')

# Закрашиваем область под линией до максимального значения
ax.fill_between(range(0, (NGEN+1)*2, 2), mean, 0, color='green', alpha=0.5)
ax.fill_between(range(0, (NGEN+1)*2, 2), 0, surface, color='blue', alpha=0.2)

ax.set_xlim(1, NGEN)
ax.set_ylim(0, 1)
ax.legend()
plt.show()


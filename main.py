import random
import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools

creator.create("FitnessMax", base.Fitness, weights=(0.5,))

creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 10
toolbox = base.Toolbox()

toolbox.register("attr_float", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n=10)
print(pop)


def evaluate(individual):
    fitness = sum(individual)
    return fitness / IND_SIZE,


toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

NGEN = 10
CXPB = 0.7
MUTPB = 0.8

s = tools.Statistics(key=lambda ind: ind.fitness.values)
s.register("mean", np.mean)
s.register("max", np.max)

hof = tools.HallOfFame(1)  # pamatuje si 1 nejlepšího jedince za historii evoluce (i když zanikne)

pop = toolbox.population(n=10)

finalpop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=s, halloffame=hof)

mean, maximum = logbook.select("mean", "max")

print(hof)

fig, ax = plt.subplots()

ax.plot(range(0, (NGEN+1)*2, 2), mean, label="mean")
ax.plot(range(0, (NGEN+1)*2, 2), maximum, label="max")
ax.set_xlim(1, NGEN)
ax.set_ylim(0, 1)
ax.legend()
plt.show()

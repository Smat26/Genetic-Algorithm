
import random
import matplotlib.pyplot as plt
from operator import add


def initializePop(popsize, xmin, xmax, ymin, ymax):
    # pop = [([(random.random() * 10) - 5 ] [(random.random() * y)])for j in range(popsize)]
    # pop = [[random.randint(-5,5)  for i in range(2)] for j in range(popsize)]
    pop = [((random.random() * (xmax - xmin)) + xmin,
            (random.random() * (ymax - ymin)) + ymin)for j in range(popsize)]
    return pop


def mutate(chromosome, xmin, xmax, ymin, ymax):
    # mutation function
    # mutation_counter = mutation_counter + 1;
    # print "mutated"

    if(random.random() > 0.5):
        chromosome[0] = chromosome[0] + 0.5
    else:
        chromosome[0] = chromosome[0] - 0.5

    if(random.random() > 0.5):
        chromosome[1] = chromosome[1] + 0.5
    else:
        chromosome[1] = chromosome[1] - 0.5

    if(chromosome[0] > xmax):
        chromosome[0] = xmax
    if(chromosome[0] < xmin):
        chromosome[0] = xmin

    if(chromosome[1] > ymax):
        chromosome[1] = ymax
    if(chromosome[1] < ymin):
        chromosome[1] = ymin
    return chromosome


def crossOver(chromosome1, chromosome2, mutation, stats, xmin, xmax, ymin, ymax):
    # crossover function
    childern = []
    stats['CrossOvers'] = stats['CrossOvers'] + 1
    child1 = [chromosome1[0], chromosome2[1]]
    child2 = [chromosome2[0], chromosome1[1]]
    childern.append(child1)
    childern.append(child2)
    if(random.random() < mutation):
        stats['mutations'] = stats['mutations'] + 1
        child1 = mutate(child1, xmin, xmax, ymin, ymax)
    if(random.random() < mutation):
        stats['mutations'] = stats['mutations'] + 1
        child2 = mutate(child2, xmin, xmax, ymin, ymax)

    return childern


def fitness(chromosome):
    # fitness criteria
    fit = chromosome[0] * chromosome[0] + chromosome[1] * chromosome[1]
    return fit


def fitness2(chromosome):
    fit = 100 * ((chromosome[0] * chromosome[0]) - chromosome[1]
                 ) * ((chromosome[0] * chromosome[0]) - chromosome[1]
                      ) + ((1 - chromosome[0]) * (1 - chromosome[0]))
    return fit


def getFitness(pop):
    # fitness provider
    fittest = []
    for p in pop:
        fittest.append(fitness2(p))
    return fittest


def getElite(pop, fittest, vip):
    elite = []
    popcopy = list(pop)
    fittestcopy = list(fittest)
    for i in range(0, vip):
        index = fittestcopy.index(max(fittestcopy))
        elite.append(popcopy[index])
        popcopy.pop(index)
        fittestcopy.pop(index)

    return elite


def nextGen(pop, fittest, vip, popsize, mutation, stats, xmin, xmax, ymin, ymax):
    parent = []
    crossed = []
    temp = getElite(pop, fittest, vip)

    # FPS(pop, fittest, vip, popsize,parent)
    binaryTournament(pop, fittest, vip, popsize,parent)


    for i in range(0, (popsize / 2) - vip - 1):
        crossed = crossOver(
            parent[i * 2], parent[i * 2 + 1], mutation, stats, xmin, xmax, ymin, ymax)
        parent[i * 2] = crossed[0]
        parent[i * 2 + 1] = crossed[1]

    for chromosome in temp:
        parent.append(chromosome)

    return parent

def FPS(pop, fittest, vip, popsize, parent):

    for i in range(0, popsize - vip):
        index = fittest.index(max(fittest))
        parent.append(pop[index])
        pop.pop(index)
        fittest.pop(index)

# need to confirm this RBS thingy
def RBS(pop, fittest, vip, popsize, parent):
    rank = [24,19,17,13,10,8,5,2,1,1];
    for i in range(0, popsize - vip):
        fittestcopy = fittest
        rank = fittestcopy.index(max(fittestcopy))

def binaryTournament(pop, fittest, vip, popsize, parent):
    for i in range(0,(popsize-vip-1), 2):
        if(fittest[i]> fittest[i+1]):
            parent.append(pop[i])
            parent.append(pop[i])
        else:
            parent.append(pop[i+1])
            parent.append(pop[i+1])



popsize = 10
vip = 0
generations = 40
mutation = 0.5
MaxFitness = []
BestSpecieX = []
BestSpecieY = []
AverageFitness = []
stats = {'mutations': 0, 'CrossOvers': 0,
         'BestSpecie': 0, 'AverageFitness': 0, 'MaxFitness': 0}


# For Problem 1
# xmin = -5
# xmax = 5
# ymin = -5
# ymax = 5


# For Problem 2
xmin = -2
xmax = 2
ymin = -1
ymax = 3

runs = 10

MaxFitnessR = [0] * generations
AverageFitnessR = [0] * generations

pop = initializePop(popsize, xmin, xmax, ymin, ymax)

for r in range(0, runs):
    for i in range(0, generations):
        fittest = getFitness(pop)
        print fittest
        stats['MaxFitness'] = max(fittest)
        MaxFitness.append(max(fittest))
        stats['BestSpecie'] = pop[fittest.index(max(fittest))]
        BestSpecieX.append(pop[fittest.index(max(fittest))][0])
        BestSpecieY.append(pop[fittest.index(max(fittest))][1])
        stats['AverageFitness'] = sum(fittest) / len(fittest)
        AverageFitness.append(sum(fittest) / len(fittest))
        pop = nextGen(pop, fittest, vip, popsize, mutation,
                      stats, xmin, xmax, ymin, ymax)
        random.shuffle(pop)
    print len(AverageFitnessR)
    print len(AverageFitness)
    for g in range(0, generations):
        AverageFitnessR[g] = AverageFitnessR[g] + AverageFitness[g]
        MaxFitnessR[g] = MaxFitnessR[g] + MaxFitness[g]

    MaxFitness = []
    AverageFitness = []

# print stats
# print fittest

MaxFitnessR[:] = [x / runs for x in MaxFitnessR]
AverageFitnessR[:] = [x / runs for x in AverageFitnessR]

plt.plot(range(0, generations), MaxFitnessR, 'r-', label='Maximum Fitness')
plt.plot(range(0, generations), AverageFitnessR, 'b-', label='Average Fitness')


# Now add the legend with some customizations.
legend = plt.legend(loc='lower center', shadow=True)

plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Binary without truncation')
plt.show()


# plt.plot(BestSpecieX, BestSpecieY, 'rx', label="Best Specie")
plt.show()
# nextGen(pop, fittest, vip, popsize)

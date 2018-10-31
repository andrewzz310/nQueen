"""""
Andrew Yiyun Zhu


In Python developed a software to solve N-Queen problem using Genetic Algorithm. 
For this purpose, I design an encoding (how to represent each state), a mutation, 
and a crossover method as well as an appropriate fitness function.

N-Queen problem:
On an NxN chess board place N queens such that all of them are safe. 
"""

import numpy as np
import sys
import random
import copy
from random import randint
#N queens (this value can be changed)
N = 8
popNum = 100
maxFitness = 0
#to hold boardState objects for the population
population = []
tempPopulation = []

'''
Class of boardStates which represents each member of the population
'''
class boardState:
    def __init__(self):
        self.chromosome = None
        self.fitness = None
        self.probability= None

    def setChromosome(self, chromosome):
        self.chromosome = chromosome
    def setFitness(self, fitness):
        self.fitness = fitness
    def setProbability(self, probability):
        self.probability = probability
    def getChromosome(self):
        return self.chromosome
    def getFitness(self):
        return self.fitness
    def getProbability(self):
        return self.probability
'''
Specifies which row queen is on from 0-N-1 starting from bottom row
and iterating from left most column to right most column
'''
def generateChromosome():
	# randomly generates a sequence of board states.
    chromo = np.arange(N)
    random.shuffle(chromo)

    return chromo
'''
Generate initial population
'''
def generatePopulation():
    #loop through the population
    for i in range (popNum):
        #generate population for each one
        population.append(boardState())
        #generate chromosome
        population[i].setChromosome(generateChromosome())
        #get fitness value
        population[i].setFitness(fitnessFunc(population[i].getChromosome()))

'''
fitness function based on what N is, for example max fitness= 28 for 8 queens,  =6 for 4 queens 
and =120 for 16 queens
'''
def fitnessFunc(chromosome):
    chromosome = chromosome
    global maxFitness
    fitness_value = 0
    cost = 0
    rowAttacks = 0
    val = N
    for val in range (N-1, 0, -1):
        fitness_value+=val

    maxFitness = fitness_value

    #find row attacks
    rowAttacks = abs(len(chromosome) - len(np.unique(chromosome)))
    cost += rowAttacks
    # find all diagonol attacks starting from left most column to right
    for column in range (N):
        for index in range (column, N):

            #only check for diagonols and not adjacent
            if (column !=index):
                diff_board = abs(column-index)
                diff_queen = abs(chromosome[column] - chromosome[index])

                if (diff_board == diff_queen):
                    cost+=1
    return fitness_value - cost
'''
Set the probability of each member getting picked as parent and also return the value of the 
highest probability
'''
def populationRankedMax(population):
    maxprobability = 0
    totalProbability = 0
    #total probability summation
    for i in range(popNum):
        totalProbability+= population[i].getFitness()
    #probability for each population is fitness value / total fitness value
    for i in range (popNum):
        population[i].setProbability(population[i].getFitness() / totalProbability)
        if (population[i].getProbability() > maxprobability):
            maxprobability = population[i].getProbability()

    return maxprobability

'''
Return the value of the minimum probability of getting picked as parent
'''
def populationRankedMin(population):
    minprobability = population[0].getProbability()
    for i in range(popNum):
        if (population[i].getProbability() < minprobability):
            minprobability = population[i].getProbability()

    return minprobability

#Select next population by first selecting the parents and then applying crossover
def nextPopulation(population, maxprobability, minprobability):
    tempParents = []
    child = boardState()
    i = 1

    while(len(tempPopulation) < popNum):
        parent1 = boardState()
        parent2 = boardState()

        # pick random value between min to max probability from the population's fitness value
        # higher fitness value will have higher chance of being selected as parent
        randomVal = random.uniform(minprobability, maxprobability)

        # select random population to be as prospective parents
        rand1 = randint(0, popNum - 1)
        rand2 = randint(0, popNum - 1)
        #only choose parents with higher than randomized probability
        while (population[rand1].getProbability() < randomVal):
            rand1 = randint(0, popNum-1)

        parent1 = population[rand1]
        tempParents.append(parent1)

        while (population[rand2].getProbability() < randomVal):
            rand2 = randint(0,popNum-1)

        parent2 = population[rand2]
        tempParents.append(parent2)

        #Perform crossover with two unique parents with higher than random probability
        if (np.array_equal(tempParents[0].getChromosome(), tempParents[1].getChromosome())):
            tempParents.clear()
        else:
            #do crossover
            child = produceCrossover(tempParents)
            tempParents.clear()
            tempPopulation.append(child)
    return tempPopulation

#add mutation for every child for one queen
def mutation(child):
    index = random.randint(0, N-1)
    child.chromosome[index] = random.randint(0, N-1)

    return child
'''
Crossover between parent and child
'''
def produceCrossover(tempParents):
    cutoffPoint = np.random.randint(N)
    child = boardState()
    child.chromosome = []
    child.chromosome.extend(tempParents[0].chromosome[0:cutoffPoint])
    child.chromosome.extend(tempParents[1].chromosome[cutoffPoint:])
    child = mutation(child)
    child.setFitness(fitnessFunc(child.getChromosome()))
    return child
'''
Get the average fitness for every iteration
'''
def averageFitness(population):
    averageFit = 0
    totalFit = 0
    for i in range(popNum):
        totalFit+=population[i].getFitness()

    averageFit = totalFit/popNum

    return averageFit
'''
Check if we have reacehd our optimal solution
'''
def completed(population, notDone):
    nDone = notDone
    for i in range(popNum):
        if (population[i].getFitness() == maxFitness):
            print("YES WE FINISHED AND FOUND SOLUTION")
            #we have finished and printing final result
            print('The solution of the final result is:')
            print(population[i].getChromosome())
            notDone = 1000000
            break
    notDone += 1
    return notDone
'''
The application method for the program
'''
if __name__ == '__main__':
    #generate initial population
    generation = 1
    maxprobability = 0
    minprobability = 0
    averageFit = 0
    notDone = 1
    generatePopulation()

    #print average fitness for first generation
    averageFit = averageFitness(population)
    maxprobability = populationRankedMax(population)
    minprobability = populationRankedMin(population)
    notDone=completed(population, notDone)
    print ('Average Fitness for this: #' + str(generation) + ' generation is: ' + str(averageFit))

    # Each subsequent Iteration/Generation
    while (notDone <=10000):
        #calculate each parents odds of getting picked
        maxprobability = populationRankedMax(population)
        minprobability = populationRankedMin(population)
        #generate the next population
        generation += 1
        tempPopulation = nextPopulation(population, maxprobability, minprobability)
        population.clear()
        population = copy.deepcopy(tempPopulation)
        averageFit = averageFitness(population)
        tempPopulation.clear()
        notDone = completed(population, notDone)
        print('Average Fitness for this: #' + str(generation) + ' generation is: ' + str(averageFit))
    #We have finished and are printing iterations
    print('Total number of iterations it took to converge to the result is: ')
    print(generation)

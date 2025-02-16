"""
This module implements the genetic algorithm for solving the snowplow routing problem.

Functions:
    similarity(S1: Solution, S2: Solution, DEPOT: int) -> int:
        Returns the similarity between two solutions. Defined as the length of the shorter route minus the number of edge sequences in common.
        Higher number means the solutions are more different. 0 means identical.
    remove_worst(population: list[Solution], beta: float) -> None:
        Removes the worst solution from the population of solutions, in place. The worst solution is indicated by having the largest cost and smallest total distance score to all other solutions.
    run_genetic(G: nx.MultiDiGraph, sp: ShortestPaths, DEPOT: int) -> Solution:
        Runs the genetic algorithm to find the best solution for snowplow routing.
"""

from .shortest_paths import ShortestPaths
from .params import POP_SIZE, BETA, N_ITER
import numpy as np
import networkx as nx
from .local_search import local_improve
from .crossover import apply_crossover
from .construction import route_generation
from .costs import routes_cost
import random
from .solution import Solution

def similarity(S1: Solution, S2: Solution, DEPOT: int) -> int:
    """
    Returns the similarity between two solutions. 
    Defined as the length of the shorter route - number of edge sequences in common
    Higher number means the solutions are more different
    0 means identical

    Args:
        S1 (Solution): the first solution to compare
        S2 (Solution): the second solution to compare
        DEPOT (int): the depot node

    Returns:
        int: measure of the similarity between the two solutions. 0 means they are identical, while a larger number means they are more different.
    """
    intersections = 0
    edge_sequences = set()
    for i in range(len(S1.route)):
        if i == 0:
            edge_sequences.add(((DEPOT, DEPOT, 0), S1.route[i]))
        elif i == len(S1.route)-1:
            edge_sequences.add((S1.route[i], (DEPOT, DEPOT, 0)))
        else:
            edge_sequences.add((S1.route[i], S1.route[i+1]))
    for i in range(len(S2.route)):
        if i == 0:
            if ((DEPOT, DEPOT, 0), S2.route[0]) in edge_sequences:
                intersections += 1
        elif i == len(S2.route)-1:
            if (S2.route[i], (DEPOT, DEPOT, 0)) in edge_sequences:
                intersections += 1
        else:
            if (S2.route[i], S2.route[i+1]) in edge_sequences:
                intersections += 1
    return len(S1.route) - intersections
    

def remove_worst(population: list[Solution], beta: float) -> None:
    """
    Removes the worst solution from the population of solutions, in place. 
    The worst solution is indicated by having the largest cost and 
    smallest total distance score to all other solutions.

    Args:
        population (list[Solution]): the population of current solutions
        beta (float): a decimal between 0 and 1, indicating how much to weigh cost vs similarity.
        This should always be greater than 0.5 so that cost is the primary consideration.
    """
    # sort costs by ascending order, so largest costs last
    costs = np.array([solution.cost for solution in population])
    temp = np.argsort(costs)
    ranks_cost = np.empty_like(temp)
    ranks_cost[temp] = np.arange(len(costs))

    # sort distances by descending order, so just take the negative. Smallest diversity last
    distances = np.array([-solution.totalSimScore for solution in population])
    temp = np.argsort(distances)
    ranks_distance = np.empty_like(temp)
    ranks_distance[temp] = np.arange(len(distances))

    ranks = beta*ranks_cost + (1-beta)*ranks_distance

    # worst = population[np.argmax(ranks)]
    del population[np.argmax(ranks)]


def run_genetic(G: nx.MultiDiGraph, sp: ShortestPaths, DEPOT: int) -> Solution:
    """
    Runs the genetic algorithm to find the optimal solution for snowplow routing.

    Args:
        G (nx.MultiDiGraph): The graph representing the road network.
        sp (ShortestPaths): The object containing the shortest paths information.
        DEPOT (int): The depot node of the graph
    Returns:
        Solution: the best solution found by the algorithm
    """
    # initialize population
    population: list[Solution] = list()
    sol_best = None
    required_edges = set(edge[:3] for edge in G.edges(data=True, keys=True) if edge[3]['priority'] != 0)

    # nearest_neighbors = sp.compute_nearest_neighbors()

    for i in range(POP_SIZE):
        print("initial generation", i)
        r, rreq = route_generation(G, sp, DEPOT)
        new_sol = Solution(rreq, dict(), routes_cost(G, sp, rreq, DEPOT), 0)
        new_sol = local_improve(new_sol, G, sp, required_edges, DEPOT)
        if i == 0:
            sol_best = new_sol
        else:
            # update similarities
            for i in range(len(population)):
                sim = similarity(population[i], new_sol, DEPOT)
                population[i].add_similarity(new_sol, sim)
                new_sol.add_similarity(population[i], sim)
            population.append(new_sol)

            if sol_best.cost > new_sol.cost:
                sol_best = new_sol


    for i in range(N_ITER):
        print("Iteration", i)
        # select a random solution
        S1 = random.choice(population)
        # select another random solution
        S2 = random.choice(list(S1.similarities.keys()))
        # apply crossover to generate new solution
        routes0 = apply_crossover(G, sp, S1.route, S2.route, DEPOT)

        new_cost = routes_cost(G, sp, routes0, DEPOT)
        new_sol = Solution(routes0, dict(), new_cost, 0)
        new_sol = local_improve(new_sol, G, sp, required_edges, DEPOT)

        # update similarities
        for i in range(len(population)):
            sim = similarity(population[i], new_sol, DEPOT)
            population[i].add_similarity(new_sol, sim)
            new_sol.add_similarity(population[i], sim)
        population.append(new_sol)

        if sol_best.cost > new_sol.cost:
            print("New best solution found")
            sol_best = new_sol
        # remove worst solution
        remove_worst(population, BETA)

    return sol_best


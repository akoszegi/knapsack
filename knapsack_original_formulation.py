import pandas as pd
import sys
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite
from math import log, ceil
import dimod
from dimod.serialization import coo
import json
import pickle

''' 
This is an implementation of Andrew Lucas' original knapsack formulation from 
https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full#h6
'''


def knapsack_bqm(costs, weights, weight_capacity):

    costs = costs

    # Initialize BQM - use large-capacity BQM so that the problem can be
    # scaled by the user.
    bqm = dimod.AdjVectorBQM(dimod.Vartype.BINARY)

    # Lagrangian multiplier (A in Lucas's paper)
    #   Note: 0 < B*max(c_a) < A
    #         0.2 <= B <= 0.5 seems like a good range
    lagrange = max(costs)  
    B = 0.5

    # Number of objects
    x_size = len(costs)

    # Lucas's algorithm introduces additional slack variables to handle
    # the inequality. In Lucas' paper y = [1, ..., W_max]
    y = [n for n in range(1, weight_capacity + 1)]
    max_y_index = len(y)
    

    # Hamiltonian xi-xi terms
    for k in range(x_size):
        bqm.set_linear('x' + str(k), (lagrange * weights[k]**2) - (B * costs[k]))

    # Hamiltonian xi-xj terms
    for i in range(x_size):
        for j in range(i + 1, x_size):
            key = ('x' + str(i), 'x' + str(j))
            bqm.quadratic[key] = 2 * lagrange * weights[i] * weights[j]

    # Hamiltonian y-y terms
    for k in range(max_y_index):
        bqm.set_linear('y' + str(k), lagrange * (y[k]**2 - 1)) 

    # Hamiltonian yi-yj terms
    for i in range(max_y_index):
        for j in range(i + 1, max_y_index):
            key = ('y' + str(i), 'y' + str(j))
            bqm.quadratic[key] = 2 * lagrange * (y[i] * y[j] + 1) 

    # Hamiltonian x-y terms
    for i in range(x_size):
        for j in range(max_y_index):
            key = ('x' + str(i), 'y' + str(j))
            bqm.quadratic[key] = -2 * lagrange * weights[i] * y[j]

    return bqm


if __name__ == "__main__":
    data_file_name = sys.argv[1] if len(sys.argv) > 1 else "data/small.csv"
    weight_capacity = float(sys.argv[2]) if len(sys.argv) > 2 else 50

    # parse input data
    df = pd.read_csv(data_file_name, header=None)
    df.columns = ['cost', 'weight']

    bqm = knapsack_bqm(df['cost'], df['weight'], weight_capacity)

    sampler = LeapHybridSampler(solver='hybrid_binary_quadratic_model_version2')
    sampleset = sampler.sample(bqm, time_limit=10)
    print(sampleset)

    # sampler = EmbeddingComposite(DWaveSampler())
    # sampleset = sampler.sample(bqm, num_reads=100)

    for sample, energy in zip(sampleset.record.sample, sampleset.record.energy):

        # Build solution from returned bitstring
        solution = []
        # energy = sampleset.record.energy[0]
        for this_bit_index, this_bit in enumerate(sample):
        # for this_bit_index, this_bit in enumerate(sampleset.record.sample[0]):
            # The x's indicate whether each object has been selected
            this_var = sampleset.variables[this_bit_index]
            if this_bit and this_var.startswith('x'):
                # Indexing of the weights is different than the bitstring;
                # cannot guarantee any ordering of the bitstring, but the
                # weights are numerically sorted
                solution.append(df['weight'][int(this_var[1:])])
        print("Found solution {} at energy {}.".format(solution, energy))

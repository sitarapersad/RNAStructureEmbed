import numpy as np
import pandas as pd
import logging as log

def simulate_modification(structure, sequence, DMS_prob, n_molecules):
    '''
    Simulate the DMS modification to a given RNA structure, producing the resulting DMS modified bases
    @param: structure- dot/bracket notation indication an open/closed base
    @param: sequence - nucleotide sequence corresponding to the given RNA structure
    @param: DMS_prob - probability of mutating an open (A/C) nucleotide
    @param: n_molecules - number of molecules to simulate
    
    @return: molecules - an array of DMS and RT'ed molecules, eg ref=ATCG, mol=ATGG
    @return: reads - an array of molecules where 0 if no mutation, 1 if mutated, eg 0010
    @return: basevectors - an array of molecules where 0 if no mutation, otherwise the inserted base is present, eg 00G0
    '''
    
    structure=list(structure)
    sequence=list(sequence)
    assert len(structure)==len(sequence), 'Sequence and structure must have the same length'
    
    molecules = []
    reads = []
    basevectors = []
    for mol_num in range(n_molecules):
        molecule = []
        read = []
        basevector = []
        for (i,pos) in enumerate(structure):
            base = sequence[i]
            if base in ['A','C'] and pos == '.': #pos=='.' indicates the base is open 
                # For a candidate base, it is DMS modified with a given probability
                if np.random.random()<DMS_prob:
                    mutation = mutate_base(base)
                    molecule.append(mutation)
                    if mutation == base:  
                        # If mutated base is the same as the original base, we cannot detect it.
                        read.append(0)
                        basevector.append('0')
                    else: 
                        read.append(1)
                        basevector.append(base)
                else:
                    molecule.append(base)
                    read.append(0)
                    basevector.append('0')
            else:
                molecule.append(base)
                read.append(0)
                basevector.append('0')
        molecules.append(molecule)
        reads.append(read)
        basevectors.append(''.join(basevector))
    
    molecules = np.array(molecules)
    reads = np.array(reads)
    
    return molecules, reads, np.array(basevectors)
                

def mutate_base(original):
    ''' 
    Mutate a DMS modified base according to RT enzyme
    @param: original - original base
    @return: choice - modified base
    '''
    
    # Naive mutation distribution, estimate better from data 
                # A      T     C    G
    mut_dist = [[0.25, 0.25, 0.25, 0.25], #A
                [0.25, 0.25, 0.25, 0.25]] #C
    
    bases = {'A':0, 'C':1}
    prob_dist = mut_dist[bases[original]]
    choice = np.random.choice(['A','T','C','G'], p=prob_dist)
    return choice

def illegal_reads(reads):
    '''
    Compute the proportion of reads which are illegal (have two mutations within distance 3 of each other)
    @param: reads - array of n bitvectors (1 is mut, 0 is WT)
    @return: illegal - array of length n, where illegal[i] is 1 if read i is illegal, otherwise 0
    '''
    def check_read(read):
        dist_3 = read+np.concatenate(([0],read[:-1]))+np.concatenate(([0,0],read[:-2]))
        if 2 in dist_3:
            return 1
        return 0
    illegal = np.apply_along_axis(check_read,1,reads)
    return illegal
import numpy as np
import pandas as pd

def naive_duplication(molecules, max_molecules):
    '''
    Naively simulate PCR by randomly sampling a molecule from the list of molecules and duplicating this molecule.Naively simulate PCR by randomly sampling a molecule from the list of molecules and duplicating this molecule.
    @param: molecules - list of molecules, where each molecule is the Query_name corresponding to an original molecule in the pool to be amplified
    @param: max_molecules - desired total number of molecules after PCR
    
    @return: molecules -  a numpy array of all the molecule query names after PCR''' 
    
    indices = [i for i in range(len(molecules))]
    while len(indices)<max_molecules:
        dup = np.random.choice(indices)
        indices.append(dup)

    indices.sort()
    return molecules[indices]
    
def naive_PCR(basevector_fname, max_molecules):
    '''
    Naively simulate PCR by randomly sampling a molecule from the list of molecules and duplicating this molecule.
    Repeat this naive amplification until we have the desired number of molecules.
    @param: basevector_fname - name of the basevector file containing molecules you wish to amplify
    @param: max_molecules - desired total number of molecules after PCR
    
    @param: molecules -  a numpy array of all the molecules after PCR'''
     
    
    df = pd.read_csv(basevector_fname, sep='\t',index_col='Query_name')
    display(df.head())
    
    N_occur = list(df['N_occur'])
    molecules = np.repeat(list(df.index), N_occur)
    print('Loaded {0} molecules.'.format(len(molecules)))

    proportions_before = N_occur/np.sum(N_occur)
    
    amplified_molecules = naive_duplication(molecules, max_molecules)
    query_names, amplified_counts = np.unique(amplified_molecules,return_counts=True)
    
    proportions_after = amplified_counts/np.sum(amplified_counts)
    
    # Plot the proportions before and after to double check that the amplification is reasonably uniform
    #TODO
    
    # Create a dictionary with a N_occur column in dataframe to contain the amplified counts
    N_df = pd.DataFrame({'Query_name':query_names,'N_occur':amplified_counts})
    N_df.set_index('Query_name', inplace=True)
    
    # Merge original datafram with new counts
    df.drop(['N_occur'],axis=1,inplace=True)
    
    df = pd.merge(df, N_df, left_index=True, right_index=True)
    df = df[['Bases_vector','Molecules','N_occur','N_mutations','N_deletions','Coverage','Reference','Index']]
    display(df.head())
    
    df.to_csv(basevector_fname[:-4]+'_PCR.txt',sep='\t')
    
    return df

   
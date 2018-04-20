import numpy as np
import pandas as pd

'''
## Fragmentation & Size Selection

Before sequencing the PCR products, they must be fragmented and size selected to a certain range. 
We simulate the fragmentation of PCR products by choosing two (?) random points to break up the PCR product and then 
select all resulting fragments within the appropriate size.

We keep track of the Query_name of the original PCR product which generated the fragment (not the original molecule, 
since we don't care which copy each fragment belongs to). All molecules that have identical modifications are given the same Query_name. 
'''

def fragment_random(reads_fname, molecule_fname, num_frags, size_range, paired_end=True, seq_length=300):
    '''
    Simulate random fragmentation on each molecule before sequencing. Generate a bitvector file of fragments as well as a 'FASTA' file.
    In the case of producing paired end reads, generates one bitvector file and two fasta files (one for each direction).
    Fasta file format example:
    >SRR041655.1 HWI-EAS284_61BKE:6:1:2:1735/1
    NAAATCAGACAAATCTCCGTTATTGGTATATACTTTGGGAGTGTTATGGAATTGCACACCCATTTCGAACATGAAGCCAATTCGTTTCTTAGGAATCGCT.
    
    @param: reads_fname - name of file containing DataFrame of reads
    @param: num_frags - number of random cuts to make per molecule
    @param: size_range -  tuple (smallest, biggest) defining the range of PCR fragments to select for 'sequencing'
    @param: paired_end - True if reads are to be sequenced using paired end technology
    @param: seq_length - Maximum length that the sequencer can read. Used only when paired_end is True, otherwise assume that size_range is chosen to capture the region of interest.
    
    @output: *_frag.csv file containing the basevectors for the fragmented regions
    @output: *.fasta - fasta file containing the products of sequencing. If paired end reads, produces two corresponding fasta files.
    
    @return: None
    '''
    df = pd.read_csv(reads_fname, sep='\t',index_col='Query_name')
    display(df.head(10))
    
    N_occur = list(df['N_occur'])
    n_basevectors = np.sum(N_occur)
    basevectors = np.repeat(list(df.Bases_vector), N_occur)
    molecules = np.repeat(list(df.Molecules), N_occur)
    query_names = np.repeat(list(df.index), N_occur)
    print(query_names)
    mol_size = len(basevectors[0])
    
    print('Loaded {0} basevectors.'.format(len(basevectors)))
    
    # Generate random fragmentation points
    frag_points = np.random.randint(mol_size, size=[n_basevectors,num_frags])
    frag_points.sort(axis=1)
    end_points = np.array([mol_size]*n_basevectors).reshape(-1,1)
    start_points = np.array([0]*n_basevectors).reshape(-1,1)
    frag_points = np.hstack((start_points,frag_points, end_points))

    # Compute the corresponding fragment lengths for each break point
    frag_lengths = np.diff(frag_points) #for example [[l_11,l_12,l_13],[l_21,l_22,l_23],...]

    # For each molecule, perform "size selection". If a fragment is of the appropriate size, it can be sequenced.
    valid_lengths = np.logical_and(frag_lengths>=size_range[0], frag_lengths<=size_range[1]) #for example [[False,True,True],[True,False,False],...]
  
    # Keep track of the results of sequencing
    seq_query = []
    seq_ampl = []
    seq_base_vec = []
    seq_start = [] #The start of coverage of that read
    seq_end = [] #The end of coverage of that read
    
    seq_mol = [] #The original bases fragmented and size selected
    
    uncovered = '.'*mol_size
    for (i, basevector) in enumerate(basevectors):
        q_name = query_names[i]     
        valid = np.where(valid_lengths[i]==True)[0] #
        frags = frag_points[i]
        for j in valid:
            start = frags[j]
            end = frags[j+1]
            bv = uncovered[:start]+basevector[start:end]+uncovered[end:]
            seq_query.append(q_name)
            seq_base_vec.append(bv)
            seq_mol.append(molecules[i][start:end])
            seq_start.append(start)
            seq_end.append(end)
            seq_ampl.append(i)
    
    df_dict = {'Amplicon':seq_ampl,'Query':seq_query,'Molecule':seq_base_vec,'Start':seq_start,'End':seq_end}
    df = pd.DataFrame(df_dict, columns=['Amplicon','Query','Molecule','Start','End'])

    df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'count'})
    display(df.head())
    df.to_csv(reads_fname[:-4]+'_frag.csv')
    print('Saved fragmented molecules dataframe.')
    
    fasta_1, fasta_2 = ([],[])
    query_1, query_2 = ([],[])
    if paired_end:
        for i,molecule in enumerate(seq_mol):
            fasta_1.append(molecule[:seq_length])  #Sequence from the beginning of the read
            fasta_2.append(molecule[-seq_length:]) #Sequence from the end of the read
            query_1.append('{0}/1'.format(seq_query[i]))
            query_2.append('{0}/2'.format(seq_query[i]))
    with open(reads_fname[:-4]+'_1.fasta','w+') as f:
        for i, molecule in enumerate(fasta_1):
            f.write('>{}\n'.format(query_1[i]))
            f.write('{}\n'.format(molecule))
            
    with open(reads_fname[:-4]+'_2.fasta','w+') as f:
        for i, molecule in enumerate(fasta_2):
            f.write('>{}\n'.format(query_2[i]))
            f.write('{}\n'.format(molecule))
    
    print('Saved fasta files.')
            
    return None

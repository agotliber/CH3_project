class Conf:

    dir_hg19 = '../res/hg19/'
    checkpoint_dir = ''
    numSurrounding = 400  # per side of CpG i.e. total is x2
    chrArr = [str(i) for i in range(1,23)]
    chrArr.extend(['X', 'Y'])
    suffix = ''
    USE_PICKLES = False
    
#     ### YOUR SETTINGS - START ###
  
# For general training and testing with 2K distances     
#     PATH = '../full/full_2k_distances/'
#     filename_sequence = 'seq.csv'
#     filename_expression = 'exp.csv'
#     filename_dist = 'dist.csv'
#     filename_labels = 'labels.csv'


# For training without testing, with train data (where test data was seperated from advance by files, for delution test) 
    PATH = "../full/full_2k_distances/train_test_split/"
    filename_sequence = 'train_seq.csv'
    filename_expression = 'train_exp.csv'
    filename_dist = 'train_dist.csv'
    filename_labels = 'train_labels.csv'
    
    validation_portion_subjects = 0.1 # 0.1
    validation_portion_probes = 0.1 # 0.1 
    train_portion_probes = 0.8 # 0.8

    ### YOUR SETTINGS - END ###

    # Below conf files are intended for use ONLY in dataProcessor, not in model code
    probeToSurroundingSeqFilePrefixAll = '../res/probe_to_surroundingSeq_'
    probeToSurroundingSeqFilePrefixChr = '../res/interims/probe_to_surroundingSeq_'
    probeToOneHotMtrxFilePrefixChr = '../res/probeToOneHotMtrx_'
    probeToOneHotMtrxFilePrefixAll = '../res/probeToOneHotMtrxAll'+str(suffix)
    probeToOneHotPrefixAll = '../res/probeToOneHotAll'+str(suffix)
    probeToOneHotPrefixChr = '../res/probeToOneHotChr_'+str(suffix)
    numBases = 5
    dfDistances = '../res/distances.csv'
    dfMethylName = 'combined_CH3'
    dfMethyl = '../res/BRCA_CA_normal_methyl.csv'
    dfExpression = '../res/BRCA_CA_normal_expressi.csv'

    numSampleInputCpgs = 4
    numInputCpgs = 5000

    epochs = 2
    num_steps = 50000
    batch_size = 512
    
    exp_n_genes = 0 #17996
    dist_n_genes = 0 #17996
    
    max_iterations_without_improvement = 5#10  # 500
    min_delta = 0.0001
    save_model_th = 0.8
    exp_th=0
    load_model = False
#     model_path = "../full/full_2k_distances/train_test_split/08_17_18_20_45/ch3 full model training_final.pkl"
    
class ConfSample(Conf):

    numSurrounding = 400 #per side
    suffix = ''
#     PATH =  '../sampled/sampled_12_04/'
#     filename_sequence = PATH + 'sampled_seq.csv'
#     filename_expression = PATH + 'sampled_exp.csv'
#     filename_dist = PATH + 'sampled_dist.csv'
#     filename_labels = PATH + 'sampled_labels.csv'
    
    USE_PICKLES = False 
    PATH =  '../sampled/mini/'
    filename_sequence = 'probeToOneHotAll_sample_mini.csv'
    filename_expression =  'e_sample_mini.csv'
    filename_dist = 'd_sample_mini.csv'
    filename_labels = 'ch3_sample_mini.csv'

#     PATH =  '../sampled/sam/'
#     filename_sequence = 'sampled_seq.csv'
#     filename_expression = 'sampled_exp.csv'
#     filename_dist = 'sampled_dist.csv'
#     filename_labels = 'sampled_labels.csv'

    validation_portion_subjects = 0.1
    validation_portion_probes = 0.1
    train_portion_probes = 0.8

    probeToOneHotPrefixAll = '../res/probeToOneHotAll_sample' + str(suffix)
    numBases = 5 #4
    dfDistances = '../res/distances_sample_withDist_10k_closest_gene.csv'

    numSampleInputCpgs = 4

    epochs = 3
    batch_size = 7
    exp_n_genes = 0 #998
    dist_n_genes = 0 #1822
    
    
sample = False
if sample:
    Conf = ConfSample
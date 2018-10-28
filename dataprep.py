import scipy.io as sio
import pandas as pd

def definitions():
    actionset = ['a'+str(i) for i in range(1,28)]
    subjectset = ['s'+str(i) for i in range(1,9)]
    repset = ['t'+str(i) for i in range(1,5)]
    
    dataset = [  '_'.join([a,s,t]) for s in subjectset for a in actionset for t in repset]
    # Remove missing elements (which are corrupted)
    dataset.remove('a8_s1_t4')
    dataset.remove('a23_s6_t4')
    dataset.remove('a27_s8_t4')

    #trainset = [ d for d in dataset if d.split('_')[1] in 's1s3s5s7']
    #validationset = [  '_'.join([a,s,t]) for s in subjectset for a in actionset for t in repset if s in 's2s4']
    #testset = [ d for d in dataset if d.split('_')[1] in 's2s4s6s8']
    
    # Implement k-fold cross-validation
    trainingsubjects = [ ''.join([ ('s'+str(i+k+1)) for i in range(4)]) for k in range(5) ]
    validationsets, trainsets = [], []
    for i in range(5):
        #validationset = [ s for s in trainset if s.split('_')[1] in ('s'+str(i*2+1)) ]
        #validationsets.append(validationset)
        #trainsets.append([s for s in trainset if s not in validationset])
        trainsets.append([ s for s in dataset if s.split('_')[1] in trainingsubjects[i] ])
        validationsets.append([ s for s in dataset if s.split('_')[1] not in trainingsubjects[i] ])
    
    return dataset, trainsets, validationsets
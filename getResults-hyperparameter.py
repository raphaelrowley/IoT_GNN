from configuration import *


dataset = 'NF-BoT-IoT' 
version = 1
randomized = True
model_type = 'E_GraphSAGE'
multiclass = True
numEpochs = 2000
numRealizations = 1
numK = [2]
dimH = [64]

data_path = os.path.join(os.getcwd(), 'hyperparam') 

file_name_f1 = f'f1_{dataset}_v{version}_{"randomized" if randomized else ""}_{model_type}_{"multiclass" if multiclass else "binary"}'
file_name_re = f're_{dataset}_v{version}_{"randomized" if randomized else ""}_{model_type}_{"multiclass" if multiclass else "binary"}'

f1 = np.zeros((len(numK), len(dimH), numRealizations))
re = np.zeros((len(numK), len(dimH), numRealizations))

for g in range(numRealizations):
    f1[:,:,g] = np.load(os.path.join(data_path, file_name_f1 + f'_g{g}.npy'))
    re[:,:,g] = np.load(os.path.join(data_path, file_name_re + f'_g{g}.npy'))

meanF1 = np.mean(f1, axis=2); meanRE = np.mean(re, axis=2)
stdF1 = np.std (f1, axis=2); stdRE = np.std(re, axis=2)
minF1 = np.min (f1, axis=2); minRE = np.min(re, axis=2)
maxF1 = np.max (f1, axis=2); maxRE = np.max(re, axis=2)

for idx_k, k in enumerate(numK):
    for idx_h, h in enumerate(dimH):
        print ("################################")
        print(f'K={k}, H={h} =>')
        print(f'F1-score: Mean={meanF1[idx_k,idx_h]:.2f}, Std={stdF1[idx_k,idx_h]:.2f}, Min={minF1[idx_k,idx_h]:.2f}, Max={maxF1[idx_k,idx_h]:.2f}')
        print(f'Recall:   Mean={meanRE[idx_k,idx_h]:.2f}, Std={stdRE[idx_k,idx_h]:.2f}, Min={minRE[idx_k,idx_h]:.2f}, Max={maxRE[idx_k,idx_h]:.2f}')
        print ("################################")
            



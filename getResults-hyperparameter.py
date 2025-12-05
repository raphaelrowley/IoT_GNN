from configuration import *


dataset = 'NF-BoT-IoT' 
version = 1
randomized = [True, False]
model_type = 'E_GraphSAGE'
multiclass = True
numEpochs = 2000
numRealizations = 5
numK = [2,3]
dimH = [64,128,256]

data_path = os.path.join(os.getcwd(), 'hyperparam') 

f1 = np.zeros((len(numK), len(dimH), numRealizations, len(randomized)))
re = np.zeros((len(numK), len(dimH), numRealizations, len(randomized)))

for g in range(numRealizations):
    for r_idx, r in enumerate(randomized):
        file_name_f1 = f'f1_{dataset}_v{version}_{"randomized" if r else ""}_{model_type}_{"multiclass" if multiclass else "binary"}'
        file_name_re = f're_{dataset}_v{version}_{"randomized" if r else ""}_{model_type}_{"multiclass" if multiclass else "binary"}'
        f1[:,:,g,r_idx] = np.load(os.path.join(data_path, file_name_f1 + f'_g{g}.npy'))
        re[:,:,g,r_idx] = np.load(os.path.join(data_path, file_name_re + f'_g{g}.npy'))

meanF1 = np.round(np.mean(f1, axis=2),2); meanRE = np.round(np.mean(re, axis=2),2)
stdF1 =  np.round(np.std (f1, axis=2),2); stdRE =  np.round(np.std (re, axis=2),2)
minF1 =  np.round(np.min (f1, axis=2),2); minRE =  np.round(np.min (re, axis=2),2)
maxF1 =  np.round(np.max (f1, axis=2),2); maxRE =  np.round(np.max (re, axis=2),2)

for idx_r, r in enumerate(randomized):
    print(f'===== Results for {"randomized" if r else "non-randomized"} dataset =====')
    for idx_k, k in enumerate(numK):
        for idx_h, h in enumerate(dimH):
            print ("################################")
            print(f'K={k}, H={h} =>')
            print(f'F1-score: Mean={meanF1[idx_k,idx_h,idx_r]}, Std={stdF1[idx_k,idx_h,idx_r]}, Min={minF1[idx_k,idx_h,idx_r]}, Max={maxF1[idx_k,idx_h,idx_r]}')
            print(f'Recall:   Mean={meanRE[idx_k,idx_h,idx_r]}, Std={stdRE[idx_k,idx_h,idx_r]}, Min={minRE[idx_k,idx_h,idx_r]}, Max={maxRE[idx_k,idx_h,idx_r]}')
            print ("################################")
            



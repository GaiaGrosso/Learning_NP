import h5py
import sys
import numpy as np
import argparse

def NP2_gen(size, seed):
    if size>10000:
        raise Warning('Sample size is grater than 100: Generator will not approximate the tale well')
    sample = np.array([])
    #normalization factor
    np.random.seed(seed)
    Norm = 256.*0.25*0.25*np.exp(-2)
    while(len(sample)<size):
        x = np.random.uniform(0,1) #assuming not to generate more than 10 000 events
        p = np.random.uniform(0, Norm)
        if p<= 256.*x*x*np.exp(-8.*x):
            sample = np.append(sample, x)
    return sample

if __name__ == '__main__':

	parser = argparse.ArgumentParser() 
	parser.add_argument('-t','--toys', type=str, help="number of toys", required=True)
	parser.add_argument('-o','--output', type=str, help="output  EOS directory", required=True)
	parser.add_argument('--joblabel', type=str, default = "LCDgun", help="job label")
	parser.add_argument('-s', '--seed', type=int, help='Specify seed', required=True)

	args = parser.parse_args()
	
	#set the random seed
	np.random.seed(int(args.seed))
	
        #whether flat distribution
        FLAT=True
	
	#choose the case of new physics
	#NP = 0 no new physics signal
	#NP = 1 peak in the tail
	#NP = 2 excess in the tail
	#NP = 3 peak in the bulk
	NP = 1 #0, 1, 2, 3
	
	NevBkg = 2000
	NevtMC = 20000
	NevSig = 0
	
	if NP == 0:
		NevSig = 0
	elif NP == 1:
		NevSig = 10
	elif NP == 2:
		NevSig = 90
	elif NP == 3:
		NevSig = 35
        else:
            raise Exception('You must specify NP variable between 0 and 3!')
	

	for i in range(int(args.toys)):
                                                                                                        
           
            if NP==0:
                if FLAT:
		    	# reference
                    	featureMC = np.random.rand(NevtMC,)
		    	# data sample
                    	featureData = np.random.rand(NevSig+NevBkg,)
                else:
			# reference
                    	featureMC = np.random.exponential(scale=0.125, size=NevtMC)
                    	# data sample                                                                                          
                    	featureData = np.random.exponential(scale=0.125, size=NevBkg)
            elif NP==1:
                if FLAT:
			# reference
                    	featureMC = np.random.rand(NevtMC,)
			# data sample
                    	featureData = np.random.rand(NevSig+NevBkg,)
                else:
                    	# reference                                                                                            
                    	featureMC = np.random.exponential(scale=0.125, size=NevtMC)
                    	# data sample
                    	featureData = np.random.exponential(scale=0.125, size=NevBkg)
                    	featureSig = np.random.normal(loc=0.8, scale=0.02, size=NevSig)
                    	featureData = np.concatenate((featureData, featureSig), axis =0)
		
            elif NP==2:
                if FLAT:
			# reference
                    	featureMC = np.random.rand(NevtMC,)
			# data sample
                    	featureData = np.random.rand(NevSig+NevBkg,)
                else:
                    	# reference                                                                                            
                    	featureMC = np.random.exponential(scale=0.125, size=NevtMC)
                    	# data sample 
                    	featureData = np.random.exponential(scale=0.125, size=NevBkg)
                    	featureSig = NP2_gen(size =NevSig, seed=int(args.seed))
                    	featureData = np.concatenate((featureData, featureSig), axis =0)
			
            elif NP==3:
                if FLAT:
			# reference
                    	featureMC = np.random.rand(NevtMC,)
                    	# data sample
			featureData = np.random.rand(NevSig+NevBkg,)
                else:
                    	# reference                                                                                            
                    	featureMC = np.random.exponential(scale=0.125, size=NevtMC)
                    	# data sample
                    	featureData = np.random.exponential(scale=0.125, size=NevBkg)
                    	featureSig = np.random.normal(loc=0.2, scale=0.02, size=NevSig)
                    	featureData = np.concatenate((featureData, featureSig), axis =0)
            


            f = h5py.File("%s/%s_%i.h5" %(args.output, args.joblabel, i))
            f.create_dataset('featureData', data=featureData, compression='gzip')
            f.create_dataset('featureMC', data=featureMC, compression='gzip')
            f.close()


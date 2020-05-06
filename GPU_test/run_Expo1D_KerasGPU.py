import os
import argparse
import numpy as np
import glob
import os.path
import time

if __name__ == '__main__':
    MEM      = 1000 #Mb
    epochs   = 1000
    DIM      = 1
    N_ref    = [1000000]
    ARC      = ['10_']
    #file_log = '/afs/cern.ch/user/g/ggrosso/work/PhD/BSM/GPU_V100_multiple_tests_log.txt'
    file_log = '/afs/cern.ch/user/g/ggrosso/work/PhD/BSM/GPU_V100_tests_log.txt'

    for (n, a) in zip(N_ref, ARC):
        #os.system("python work/PhD/BSM/NOTEBOOKS/Expo1D_KerasGPU-AGGREGATOR.py %i %i %s %i %i %s"%(n, DIM, a, epochs, MEM, file_log))
        os.system("python work/PhD/BSM/NOTEBOOKS/Expo1D_KerasGPU.py %i %i %s %i %i %s"%(n, DIM, a, epochs, MEM, file_log))

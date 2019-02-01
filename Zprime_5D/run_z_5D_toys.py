import os
import argparse
import numpy as np
import glob
import os.path
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
    parser.add_argument('-o','--output', type=str, help="output  EOS directory", required=True)
    parser.add_argument('-i','--input', type=str, help="input  EOS directory", required=True)
    parser.add_argument('-p','--pyscript', type=str,default = "TOY2.py",  help="name of python script to execute")
    parser.add_argument('-q', '--queue', type=str, default = "1nw", help="LSFBATCH queue name")
#    parser.add_argument('-f', '--feature', type=str, default = "0", help="feature of the input daatset to be analyzed")
    parser.add_argument('-t', '--toys', type=str, default = "1000", help="number of toys to be processed")
    args = parser.parse_args()
    
    #folder to save the outputs of the pyscript
    mydir = args.output+"/"
    os.system("mkdir %s" %mydir)
    
    #folder to save the outputs of each condor job (file.out, file.log, file.err)
    label = args.output.split("/")[-1]+'_5D_'+str(time.time())
    os.system("mkdir %s" %label)

    for i in range(int(args.toys)):
        joblabel = str(i)#fileIN.split("/")[-1].replace(".h5","")                                                                                                                      
        if not os.path.isfile("%s/%s_t.txt" %(mydir, joblabel)):
            # src file
            script_src = open("%s/%s.src" %(label, joblabel) , 'w')
            script_src.write("#!/bin/bash\n")
            script_src.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_92//x86_64-slc6-gcc62-opt/setup.sh\n")
            script_src.write("python %s/%s %s %s %s" %(os.getcwd(),args.pyscript, mydir, joblabel, args.input))
            script_src.close()
            os.system("chmod a+x %s/%s.src" %(label, joblabel))
            #os.system("bsub -q %s -o %s/%s.log -J %s_%s < %s/%s.src" %(args.queue, label, joblabel, label, joblabel, label, joblabel))
            
            # condor file
            script_condor = open("%s/%s.condor" %(label, joblabel) , 'w')
            script_condor.write("executable = %s/%s.src\n" %(label, joblabel))
            script_condor.write("universe = vanilla\n")
            script_condor.write("output = %s/%s.out\n" %(label, joblabel))
            script_condor.write("error =  %s/%s.err\n" %(label, joblabel))
            script_condor.write("log = %s/%s.log\n" %(label, joblabel))
            script_condor.write("+MaxRuntime = 500000\n")
            script_condor.write("queue\n")
            script_condor.close()
            # condor file submission
            os.system("condor_submit %s/%s.condor" %(label, joblabel))

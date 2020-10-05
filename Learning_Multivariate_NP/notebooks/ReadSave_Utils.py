# python libraries
import glob
import numpy as np
import h5py
import math
import time
import matplotlib.pyplot as plt
import os
import getpass

def collect_t(DIR_IN, DIR_OUT='None'):
    '''
    For each toy the function reads the .txt file where the final value for the variable t=-2*loss is saved. 
    It then associates a label to each toy.
    The array of the t values (tvalues) and the array of labels (files_id) are saved in an .h5 file.
    
    DIR_IN: directory where all the toys' outputs are saved
    DIR_OUT: directory where to save the .h5 output file
    
    The function returns the array of labels.
    '''
    dt = h5py.special_dtype(vlen=str)
    tvalues = np.array([])
    files_id = np.array([])
    FILE_TITLE=''
    for fileIN in glob.glob("%s*_t.txt" %DIR_IN):
        #print('prova')
        #print(fileIN)
        f = open(fileIN)
        lines = f.readlines()
        file_id=  fileIN.split('/')[-1]
        FILE_TITLE = fileIN.split('/')[-2]
        file_id = file_id.replace('_t.txt', '')
        f.close()
        #print(file_id)
        if len(lines)==0:
            continue
        t = float(lines[0])
        #print(file_id)
        if(np.isnan(np.array([t]))): 
            continue 
        #print(t)
        tvalues  = np.append(tvalues, t)
        files_id = np.append(files_id, file_id)
        
    files_id=np.array(files_id, dtype=dt)
    # save tvalues in a h5 file
    if DIR_OUT !='None':
        f = h5py.File(DIR_OUT+ FILE_TITLE+'_tvalues.h5', 'w')
        f.create_dataset('tvalues', data=tvalues, compression='gzip')
        f.create_dataset('files_id', data=files_id, compression='gzip')
        f.close()
    
    return files_id, FILE_TITLE
    

def collect_loss_history(files_id, DIR_IN, patience):
    '''
    For each toy whose file ID is in the array files_id, 
    the function collects the history of the loss and saves t=-2*loss at the check points.
    
    files_id: array of toy labels 
    DIR_IN: directory where all the toys' outputs are saved
    patience: interval between two check points (epochs)
    
    The function returns a 2D-array with final shape (nr toys, nr check points).
    '''
    tdistributions_check = np.array([])
    
    cnt=0
    for file_id in files_id:
        history_file = DIR_IN+file_id+'_history'+str(patience)+'.h5'
        print(history_file)
        if not os.path.exists(history_file):
            continue
        f = h5py.File(history_file, 'r')
        
        loss   = f.get("loss")
        #epoch  = f.get("epoch")
        if not loss:
            print("not")
            continue
        loss     = np.array(loss) 
        loss     = np.expand_dims(loss, axis=1)
        
        if not cnt:
            # initialize the array at the first iteration
            tdistributions_check = -2*loss
        else:
            # just append to tdistributions_check
            tdistributions_check = np.concatenate((tdistributions_check, -2*loss), axis=1)
            
        print(str(cnt)+': toy '+file_id+' loaded.')
        cnt = cnt+1
        f.close()
        #print(tdistributions_check.shape)
    print('Final history array shape')
    print('(nr toys, nr check points)')
    print(tdistributions_check.T.shape)
    return tdistributions_check.T
    
def Save_loss_history_to_h5(DIR_OUT, file_name, extension, patience, tvalues_check):
    '''
    The function save the 2D-array of the loss histories in an .h5 file.
    
    DIR_OUT: directory where to save the output file
    file_name: output file name
    extension: label to be appended to the file_name
    '''
    epochs_check = []
    nr_check_points = tvalues_check.shape[1]
    for i in range(nr_check_points):
        epoch_check = patience*(i+1)
        epochs_check.append(epoch_check)
        
    log_file = DIR_OUT+file_name+extension+'.h5' #'_tvalues_check.h5'
    print(log_file)
    f = h5py.File(log_file,"w")
    for i in range(tvalues_check.shape[1]):
        f.create_dataset(str(epochs_check[i]), data=tvalues_check[:, i], compression='gzip')
    f.close()
    print('Saved to file: ' +file_name+extension+'.h5')
    return

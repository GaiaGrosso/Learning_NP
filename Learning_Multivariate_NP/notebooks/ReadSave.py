
# input directory
DIR_INPUT = '/eos/user/.../Z_5D_Mcut100.0_PTcut20.0_ETAcut2.4_sb0_se0_eb0_ee0_patience1000_ref1000000_bkg200000_epochs300000_latent5_layers3_wclip2.15/'
if not DIR_INPUT.endswith('/'):
    DIR_INPUT=DIR_INPUT+'/'
    
title    = DIR_INPUT.split('/')[-2]
patience = DIR_INPUT.split("patience",1)[1] 
patience = patience.split("_",1)[0]

# output directory
output_path = DIR_INPUT+'out/'
#output_path='None'
if not os.path.exists(output_path) and not 'None' in output_path:
    os.makedirs(output_path)



# collect t
files_id, FILE_TITLE = collect_t(DIR_INPUT, output_path)

# collect loss history
loss_history = collect_history(files_id, DIR_INPUT, int(patience))

Save_loss_history_to_h5(output_path, title, '_loss_history', int(patience), loss_history)

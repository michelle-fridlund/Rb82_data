from CAAI.train import CNN
from CAAI.predict import CNN as TrainedModel
import pickle, os
import numpy as np
from data_generator import DataGenerator
import pyminc.volumes.factory as pyminc

def train_v1():
    
    cnn = CNN(model_name='v1',
              input_patch_shape=(128,128,16),
              input_channels=2,
              output_channels=1,
              batch_size=2,
              epochs=2000,
              learning_rate=1e-4,
              checkpoint_save_rate=50,
              loss_functions=[['mean_absolute_error',1]],
              data_pickle='/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc/data_6fold.pickle',
              data_folder='/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc',
              data_pickle_kfold=1
              )
    
    # Attach generator
    cnn.data_loader = DataGenerator(cnn.config)  

    cnn.print_config()

    final_model_name = cnn.train()    
    
    return final_model_name
    
def predict(modelh5name, model_name=None):
    
    modelbasename = os.path.splitext(os.path.basename(modelh5name))[0]
    model = TrainedModel(modelh5name)
    
    # Overwrite name if set
    if model_name:
        modelbasename = model_name
    
    summary = pickle.load( open('/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc/data_6fold.pickle', 'rb') )
    for pt in summary['valid_1']:
        predict_patient(pt,model,modelbasename)

def predict_patient(pt,model,modelbasename):
    _lowdose_name = "FDG_01_SUV.mnc"
    data_folder = '/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc'
    fname_dat = os.path.join(data_folder,pt,'dat_01_suv_ctnorm_double.npy')
    dat = np.memmap(fname_dat, dtype='double', mode='r')
    dat = dat.reshape(128,128,-1,2)
    
    print("Predicting volume for %s" % pt)
    predicted = np.empty((111,128,128))
    x = 128
    y = 128
    z = 16
    d = 2
    for z_index in range(int(z/2),111-int(z/2)):
        predicted_stack = model.predict(dat[:,:,z_index-int(z/2):z_index+int(z/2),:].reshape(1,x,y,z,d))
        if z_index == int(z/2):
            for ind in range(int(z/2)):
                predicted[ind,:,:] = predicted_stack[0,:,:,ind].reshape(128,128)
        if z_index == 111-int(z/2)-1:
            for ind in range(int(z/2)):
                predicted[z_index+ind,:,:] = predicted_stack[0,:,:,int(z/2)+ind].reshape(128,128)
        predicted[z_index,:,:] = predicted_stack[0,:,:,int(z/2)].reshape(128,128) 
    predicted_full = predicted
    predicted_full += np.swapaxes(np.swapaxes(dat[:,:,:,0],2,1),1,0)
    
    out_vol = pyminc.volumeLikeFile(os.path.join(data_folder,pt,_lowdose_name),os.path.join(data_folder,pt,'predicted_'+modelbasename+'_'+_lowdose_name))
    out_vol.data = predicted_full
    out_vol.writeFile()
    out_vol.closeVolume()

if __name__ == '__main__':
    model_name = train_v1()
    predict(model_name)
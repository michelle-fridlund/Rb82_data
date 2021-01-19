import warnings
warnings.filterwarnings('ignore')
import os
import pickle
from keras.models import load_model, model_from_json

class CNN():
    def __init__(self,model,config=None,custom_objects={}):
        
        if config:
            self.config = pickle.load(open(config,'rb'))
            model = self.generate_model_name_from_params()

        # Setup network
        if model.endswith('.json'):
            self.model = self.load_model_w_json(model)
        else:
            self.model = load_model(model,custom_objects=custom_objects)


    def load_model_w_json(self,model):
        modelh5name = os.path.join( os.path.dirname(model), os.path.splitext(os.path.basename(model))[0]+'.h5' )
        json_file = open(model,'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(modelh5name)
        return model

    def generate_model_name_from_params(self):
        # Build full model name
        model_name = self.config['model_name']

        model_name += '_e{}'.format( self.config['epochs'] )
        model_name += '_bz{}'.format( self.config['batch_size'] )
        model_name += '_lr{:.1E}'.format( self.config['learning_rate'] )
        model_name += '_DA' if self.config['augmentation']  else '_noDA'
        model_name += '_TL' if self.config['pretrained_model'] is not None else '_noTL'
        model_name += '_LOG%d' % self.config['data_pickle_kfold'] if self.config['data_pickle_kfold'] is not None else ''

        return model_name
    
    def predict(self,X):
        return self.model.predict(X)
     

        
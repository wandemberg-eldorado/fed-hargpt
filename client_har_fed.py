import sys
import flwr as fl
import har_transformer
from har_transformer import models
from har_transformer import utils
import getopt 
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

labels = ['label:LYING_DOWN',
 'label:SITTING',
 'label:FIX_walking',
 'label:FIX_running',
 'label:BICYCLING',
 'label:SLEEPING',
 'label:LAB_WORK',
 'label:IN_CLASS',
 'label:IN_A_MEETING',
 'label:LOC_main_workplace',
 'label:OR_indoors',
 'label:OR_outside',
 'label:IN_A_CAR',
 'label:ON_A_BUS',
 'label:DRIVE_-_I_M_THE_DRIVER',
 'label:DRIVE_-_I_M_A_PASSENGER',
 'label:LOC_home',
 'label:FIX_restaurant',
 'label:PHONE_IN_POCKET',
 'label:OR_exercise',
 'label:COOKING',
 'label:SHOPPING',
 'label:STROLLING',
 'label:DRINKING__ALCOHOL_',
 'label:BATHING_-_SHOWER',
 'label:CLEANING',
 'label:DOING_LAUNDRY',
 'label:WASHING_DISHES',
 'label:WATCHING_TV',
 'label:SURFING_THE_INTERNET',
 'label:AT_A_PARTY',
 'label:AT_A_BAR',
 'label:LOC_beach',
 'label:SINGING',
 'label:TALKING',
 'label:COMPUTER_WORK',
 'label:EATING',
 'label:TOILET',
 'label:GROOMING',
 'label:DRESSING',
 'label:AT_THE_GYM',
 'label:STAIRS_-_GOING_UP',
 'label:STAIRS_-_GOING_DOWN',
 'label:ELEVATOR',
 'label:OR_standing',
 'label:AT_SCHOOL',
 'label:PHONE_IN_HAND',
 'label:PHONE_IN_BAG',
 'label:PHONE_ON_TABLE',
 'label:WITH_CO-WORKERS',
 'label:WITH_FRIENDS']

# Load model and data
def run_client(df_path, model_path, user):
    
    config = {
        'data_params': {
            'labels': labels,
            'data_csv': df_path 
        },
        'transformer_params': {
            'transformers_layers': 4, 
            'n_positions': 128,
            'hidden_size': 384, 
        },

        'training_params': {
            'batch_size': 64,
            'epochs': 5000,
            'lr': 4.00e-05
        }

    }

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    
    train_dataloader, validation_dataloader, test_dataloader, input_dim = har_transformer.prep_tensor(config)
    config['data_params']['input_dim'] = input_dim
    
    har = utils.build_model(config)
    state_dict = torch.load(model_path, map_location=torch.device('cuda:0'))
    har.load_state_dict(state_dict, strict=False)

    # Start Flower client
    fl.client.start_numpy_client(client=HARClient(har, train_dataloader, validation_dataloader, test_dataloader, config, user), server_address="[::]:8080") 

    # Define Flower client
class HARClient(fl.client.NumPyClient):
    def __init__(self, har : models.HarTransformer, train_dataloader : DataLoader, validation_dataloader : DataLoader, test_dataloader : DataLoader, config : dict, user : str) -> None:
        super().__init__()
        self.har = har
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.user = user

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.har.state_dict().items()]
    
    def set_parameters(self, parameters, config):
        params_dict = zip(self.har.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.har.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        har_transformer.train(self.har, self.train_dataloader, self.validation_dataloader, self.config['device'], len(self.config['data_params']['labels']), epochs=self.config['training_params']['epochs'], lr=self.config['training_params']['lr'])
        num_examples = len(self.train_dataloader.dataset)
        return self.get_parameters(config), num_examples, {}
    
        

    def evaluate(self, parameters, config):
    
        self.set_parameters(parameters, config)
        ba, clf_report, array, test_loss = har_transformer.test(self.har, self.test_dataloader, self.config['device'], labels)

        num_examples = len(self.test_dataloader.dataset)

        torch.save(self.har.state_dict(), f'models/exp_0/model_{self.user}.pt')

        return float(test_loss), num_examples, {"accuracy": float(ba)}



if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "he:u:", ["help", "exp=", "user="])
    except:
        print('call with -h or --help to see the options')
        sys.exit(2)


    for opt, arg in opts:
        if opt in ['-h', '--help']:
            print('help')
            sys.exit(0)
        elif opt in ['-e', '--exp']:
            exp = arg
            #print('here')
        elif opt in ['-u', '--user']:
            #print('nop')
            user = arg
            #print('finally')
        

    run_client(f'data/full_data/exp_Old/fold_{exp}/{user}/{user}.features_labels.csv', f'models/model_opt.pt', user)
    sys.exit(0)

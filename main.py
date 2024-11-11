import datetime
import pickle
import numpy as np

import torch
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

from har_transformer.models import HarTransformer
from har_transformer import data_prep, train, test, create_k_folds_n_users, prep_tensor, create_unified_csv

def build_model(config: dict):
    input_dim = config['data_params']['input_dim']
    labels_dim = len(config['data_params']['labels']) #beware of data leak
    hidden_size = config['transformer_params']['hidden_size']
    n_positions = config['transformer_params']['n_positions']
    transformers_layers = config['transformer_params']['transformers_layers']

    model = HarTransformer(input_dim, labels_dim, hidden_size, transformers_layers=transformers_layers, n_positions=n_positions)
    model.to(config['device'])
    return model


def objective(trial, train_dataloader, validation_dataloader, test_dataloader, input_dim, labels, device):
    params = {
        'training_params': {
            'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'epochs': trial.suggest_categorical('epochs', [50, 100, 200, 300, 400]),
        },
        'transformer_params': {
            'hidden_size': trial.suggest_categorical('hidden_size', [48, 96, 192, 384, 768]),
            'n_positions': trial.suggest_categorical('n_positions', [32, 64, 128, 256]),
            'transformers_layers': trial.suggest_categorical('transformers_layers', [1, 2, 3, 4, 6, 12])
        },
        'data_params': {
            'input_dim': input_dim,
            'labels': labels
        },
        'device': device
    }



    labels_dim = len(config['data_params']['labels'])
    labels = config['data_params']['labels']
    model = build_model(params)
    val_ba, val_ba_array, train_loss_set = train(model, train_dataloader, validation_dataloader, device, labels_dim, epochs=params['training_params']['epochs'], lr=params['training_params']['lr'])
    ba, clf_report, ba_array, test_loss = test(model, test_dataloader, params['device'], labels)

    return ba


def hyperparameter_optimization(train_dataloader, validation_dataloader, test_dataloader, input_dim, labels, device):
    n_trials = 50
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())

    func = lambda trial: objective(trial, train_dataloader, validation_dataloader, test_dataloader, input_dim, labels, device)

    wandb_kwargs = {"project": "har_transformer_new_loss", "group": "hyperparameter_optimization"}

    wandbc = WeightsAndBiasesCallback(metric_name="balanced_accuracy", wandb_kwargs=wandb_kwargs)

    study.optimize(func, n_trials=n_trials, callbacks=[wandbc])
    
    best_trial = study.best_trial


    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    params = best_trial.params

    
    params['data_params'] =  {'input_dim': input_dim, 'labels': labels}
    params['device'] = device

    model = build_model(params)
    val_ba, val_ba_array, train_loss_set = train(model, train_dataloader, validation_dataloader, params, epochs=params['training_params']['epochs'], lr=params['training_params']['lr'])


    ba, clf_report, ba_array, test_loss = test(model, test_dataloader, params['device'], labels)
    print("val_ba: {}, test_ba: {}".format(ba, val_ba))

    pickle.dump(train_loss_set, open('train_loss_set.txt','wb'))
    pickle.dump(ba_array, open('test_ba_array.txt', 'wb'))

    return model, ba, clf_report


def main(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    
    train_dataloader, validation_dataloader, test_dataloader, input_dim = prep_tensor(config)
    config['data_params']['input_dim'] = input_dim

    labels_dim = len(config['data_params']['labels'])
    labels = config['data_params']['labels']
    
    model = build_model(config)
    train(model, train_dataloader, validation_dataloader, device, labels_dim, epochs=config['training_params']['epochs'], lr=config['training_params']['lr'])
    ba, clf_report, ba_array, test_loss = test(model, test_dataloader, device, labels)

    torch.save(model.state_dict(), 'models/model_opt.pt')
    print(clf_report)

    print(f'Test Balanced Accuracy (BA): {ba}')
    print(f'Test Balanced Accuracies (BAs): {ba_array}')
    return model, ba, clf_report

def run_with_optuna(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    
    train_dataloader, validation_dataloader, test_dataloader, input_dim = prep_tensor(config)
    config['data_params']['input_dim'] = input_dim

    labels = config['data_params']['labels']
    
    model, ba, clf_report = hyperparameter_optimization(train_dataloader, validation_dataloader, test_dataloader, input_dim, labels, device)

    print(clf_report)


    torch.save(model.state_dict(), f'models/model_{datetime.now()}.pt')

    print(clf_report)
    print(f'Test Balanced Accuracy (BA): {ba}')
    return model, ba, clf_report


#TODO: improve
def run_with_user_crossvalidation(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    
    labels = config['labels']
    
    k_folds = create_k_folds_n_users(5, 'data/full_data')


    bas = []
    for fold in k_folds:
        train_dataloader, validation_dataloader, test_dataloader, input_dim = prep_tensor(config)
        config['data_params']['input_dim'] = input_dim
        
        config = {
            'df_path': k_folds[fold]['massive_train'],
            'labels': labels
        }
        
        model, ba, clf_report = hyperparameter_optimization(train_dataloader, validation_dataloader, test_dataloader, input_dim, labels, device)
        bas.append(ba.numpy())
        
        model.save(f'model/saved_model_{fold}')

        print(clf_report)
        print(f'Test Balanced Accuracy (BA): {np.ndarray(ba).mean()}')
        print(f'All BAs: {bas}')


    print(bas)

    model.summary()


if __name__ == '__main__':
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
    
    
    config = {
        'data_params': {
            'labels': labels,
            'data_csv': 'data/full_data/exp_Old/fold_0/raw_40.csv',
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

    main(config)

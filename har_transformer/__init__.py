

import os
import glob

import pickle
from tqdm import trange

import numpy as np
import pandas as pd
from  sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, f1_score, accuracy_score

import torch
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torchmetrics.functional.classification import multilabel_confusion_matrix


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def data_prep(config: dict):
    device = config['device']
    data_csv = config['data_params']['data_csv']

    df = pd.read_csv(data_csv)
    df.fillna(0.0, inplace=True)
    df['one_hot_labels'] = list(df[config['data_params']['labels']].values)

    label_counts = df.one_hot_labels.astype(str).value_counts()
    one_freq = label_counts[label_counts==1].keys()
    one_freq_idxs = sorted(list(df[df.one_hot_labels.astype(str).isin(one_freq)].index), reverse=True)
    print('df label indices with only one instance: ', one_freq_idxs)

    df.drop(one_freq_idxs, inplace=True)
    df.drop(columns=['timestamp'], inplace=True)

    x = df[df.columns.drop(df.filter(regex='label:'))]
    x = x[x.columns.drop('one_hot_labels')]
    y = df[['one_hot_labels']]

    input_dim = x.shape[1]

    labels_enc_test = list(df.one_hot_labels.values)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, stratify=labels_enc_test)

    labels_enc_val = list(y_train.one_hot_labels.values)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2, stratify=labels_enc_val)


    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory
    x_train_tensor = torch.Tensor(x_train.values).to(device)
    x_train_tensor = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], 1, x_train_tensor.shape[1],))
    y_train_tensor = torch.Tensor(list(y_train['one_hot_labels'].values)).to(device)

    x_val_tensor = torch.Tensor(x_val.values).to(device)
    x_val_tensor = torch.reshape(x_val_tensor, (x_val_tensor.shape[0], 1, x_val_tensor.shape[1],))
    y_val_tensor = torch.Tensor(np.array(list(y_val['one_hot_labels'].values))).to(device)


    x_test_tensor = torch.Tensor(x_test.values).to(device)
    x_test_tensor = torch.reshape(x_test_tensor, (x_test_tensor.shape[0], 1, x_test_tensor.shape[1],))
    y_test_tensor = torch.Tensor(np.array(list(y_test['one_hot_labels'].values))).to(device)

    return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor, input_dim 


def prep_tensor(config):
    x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor, input_dim  = data_prep(config)
    #num_workers=6

    # Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
    batch_size = config['training_params']['batch_size']

    
    # Make tensors out of data
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_sampler = RandomSampler(train_data)
    #train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(x_val_tensor, y_val_tensor)
    validation_sampler = SequentialSampler(validation_data)
    #validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size, num_workers=num_workers)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Create test dataloader
    test_data = TensorDataset(x_test_tensor, y_test_tensor)
    test_sampler = SequentialSampler(test_data)
    #test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # Save test dataloader
    torch.save(validation_dataloader,'data/validation_data_loader')
    torch.save(train_dataloader,'data/train_data_loader')
    torch.save(test_dataloader,'data/test_data_loader')

    return train_dataloader, validation_dataloader, test_dataloader, input_dim


def train (model, train_dataloader, validation_dataloader, device, labels_dim, epochs=300, lr=2e-5):
    # setting custom optimization parameters. You may implement a scheduler here as well.
    param_optimizer = list(model.transformer.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    #optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5,correct_bias=True)
    optimizer = AdamW(optimizer_grouped_parameters,lr=lr)


    
    # Store our loss and accuracy for plotting
    train_loss_set = []

    for epoch in trange(epochs, desc="Epoch"):

        # Training
        
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0 #running loss
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            b_input_ids, b_labels = batch
            optimizer.zero_grad()

            outputs = model(b_input_ids)
            logits = outputs#[0]
            loss_func = BCEWithLogitsLoss(weight=calculate_pos_weights(b_labels, device=device)) 
            #loss_func = BCEWithLogitsLoss() 
            loss = loss_func(logits.view(-1,labels_dim),b_labels.type_as(logits).view(-1,labels_dim)) #convert labels to float for calculation
            train_loss_set.append(loss.item())    

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        ###############################################################################

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Variables to gather full output
        logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

        # Predict
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_labels = batch
            with torch.no_grad():
                # Forward pass
                outs = model(b_input_ids)
                b_logit_pred = outs#[0]
                pred_label = torch.sigmoid(b_logit_pred)  

                b_logit_pred = b_logit_pred.detach().cpu().numpy()
                pred_label = pred_label.to('cpu').numpy()
                b_labels = b_labels.to('cpu').numpy()

                tokenized_texts.append(b_input_ids)
                logit_preds.append(b_logit_pred)
                true_labels.append(b_labels)
                pred_labels.append(pred_label)


        # Flatten outputs
        pred_labels = [item for sublist in pred_labels for item in sublist]
        pred_labels = [item for sublist in pred_labels for item in sublist] #yes, repeat

        true_labels = [item for sublist in true_labels for item in sublist]

        # Calculate Accuracy
        threshold = 0.50
        pred_bools = [pl>threshold for pl in pred_labels]
        true_bools = [tl==1 for tl in true_labels]
        val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')*100
        val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100

        print('F1 Validation Accuracy: ', val_f1_accuracy)
        print('Flat Validation Accuracy: ', val_flat_accuracy)

        ba, ba_array = avg_multilabel_BA(true_bools, pred_bools, labels_dim)
        array = np.array([x.item() for x in ba_array])

        print('Flat Validation Balanced Accuracy (BA): ', ba)

        if epoch % 1000 == 0:
            checkpoint(model, f"models/epoch-{epoch}.pth")

    return ba, array, train_loss_set #, val_loss_set


def test(model, test_dataloader, device, labels):

    # Test

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    #track variables
    logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

    test_loss = 0
    # Predict
    for i, batch in enumerate(test_dataloader):
        #batch = tuple(t.to(device) for t in batch)
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_labels = batch
        with torch.no_grad():
            # Forward pass
            outs = model(b_input_ids)
            b_logit_pred = outs#[0]
            pred_label = torch.sigmoid(b_logit_pred)

            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)

    # Flatten outputs
    tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]  #yes, repeat
    true_labels = [item for sublist in true_labels for item in sublist]
    # Converting flattened binary values to boolean values
    true_bools = [tl==1 for tl in true_labels]


    pred_bools = [pl>0.50 for pl in pred_labels] #boolean output after thresholding

    # Print and save classification report
    print('Test F1 Accuracy: ', f1_score(true_bools, pred_bools,average='micro'))
    print('Test Flat Accuracy: ', accuracy_score(true_bools, pred_bools),'\n')
    clf_report = classification_report(true_bools,pred_bools,target_names=labels)
    pickle.dump(clf_report, open('classification_report.txt','wb')) #save report

    ba, ba_array = avg_multilabel_BA(true_bools, pred_bools, len(labels))
    array = np.array([x.item() for x in ba_array])

    print(f'Flat Test Balanced Accuracy (BA): {ba}')
    print(f'Complete Test Balanced Accuracy (BA): {array}')

    return ba, clf_report, array, test_loss



def avg_multilabel_BA(y_true, y_pred, num_labels):
    '''
    Calculate the Balanced Accuracy
        :param np.array y_true: true labels
        :param np.array y_pred: predicted labels
        :return: Balanced Accuracy
        :rtype: float
    '''
    ba_array = []
    
    y_pred_tensor = torch.Tensor(y_pred).to('cpu')
    y_pred_tensor = y_pred_tensor.type(torch.int32)

    y_true_tensor = torch.Tensor(y_true).to('cpu')
    y_true_tensor = y_true_tensor.type(torch.int32)

    multi_cm = multilabel_confusion_matrix(y_pred_tensor, y_true_tensor, num_labels=num_labels)

    for i in range(num_labels):
        cm = multi_cm[i]
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp + fn) #if tp + fn is 0, then there is no positive sample in this label, setting this to 1.0. BA will the be determined by specificity only

        sensitivity = torch.nan_to_num(sensitivity, nan=1.0)

        ba = 0.5*(specificity+sensitivity)

        ba_array.append(ba)

    mean_ba = np.mean(np.array(ba_array))

    return mean_ba, ba_array


def get_all_user_csvs(folderpath : str) -> list:
    '''
    Iterate over a dict and get all csv files
        :param str folderpath: path to the folder
        :return: list of csv files
        :rtype: list
    '''
    answer = []
    for filename in glob.iglob(f'{folderpath}/**.csv', recursive=True):
        answer.append(filename)
    return answer


def create_k_folds_n_users(k_folds: int, folderpath: str):
    '''
    Create k folds for cross validation
        :param k_folds: number of folds
        :param int n_users: number of users
        :param str folderpath: path to the folder
        :return: dict with the folds' base data
        :rtype: dict
    '''
    all_csvs = get_all_user_csvs(folderpath)
    all_dfs = {}
    kf = KFold(n_splits=k_folds, shuffle=True)
    
    i=0
    for train_index, test_index in kf.split(all_csvs):
        fold_list_train, fold_list_test = np.array(all_csvs)[train_index], np.array(all_csvs)[test_index]
    
        fold_df_train = pd.read_csv(fold_list_train[0])
        for csv in fold_list_train[1:]:
            fold_df_train = fold_df_train.append(pd.read_csv(csv))

        spl = csv.split(f'{folderpath}/')
        fold_list_test = np.setdiff1d(all_csvs, fold_list_train)
        fold_list_test = [spl[0] for csv in fold_list_test]

        # Path
        path_exp = os.path.join(folderpath, f'exp_/fold_{i}')
        try:
            os.mkdir(path_exp)
        except FileExistsError as e:
            pass

        for test_user in fold_list_test:
            user_id = test_user.split('.features_labels.csv')[0]

            path_user = os.path.join(path_exp, f'{user_id}')
            try:
                os.mkdir(path_user)
            except FileExistsError as e:
                pass

            raw = pd.read_csv(os.path.join(folderpath, test_user)).fillna(0.0)
            x = raw[raw.columns.drop(raw.filter(regex='label:'))]
            y = raw.filter(regex='label:')
            x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=42)
            
            x_train.to_csv(f'{path_user}/x_train.csv', index=False)
            x_test.to_csv(f'{path_user}/x_test.csv', index=False)
            y_train.to_csv(f'{path_user}/y_train.csv', index=False)
            y_test.to_csv(f'{path_user}/y_test.csv', index=False)

        fold_df_train.to_csv(f'{path_exp}/raw_massive_train.csv', index=False)
        #salvo os paths
        all_dfs[f'fold_{i}'] = {'massive_train': f'{path_exp}/raw_massive_train.csv'} 
        i+=1

    return all_dfs

def create_unified_csv(folderpath: str, path_exp: str, final_filename: str):
    '''
    Create k folds for cross validation
        :param k_folds: number of folds
        :param int n_users: number of users
        :param str folderpath: path to the folder
        :return: dict with the folds' base data
        :rtype: dict
    '''
    all_csvs = get_all_user_csvs(folderpath)
    
    i=0
    
    

    uni_df = pd.read_csv(all_csvs[0])
    for csv in all_csvs[1:]:
        uni_df = uni_df.append(pd.read_csv(csv))

    uni_df.to_csv(f'{path_exp}/{final_filename}.csv', index=False)

    return uni_df

def calculate_pos_weights(y, device: str):
    pos_weights = []
    try:
        for i in range(y.shape[1]):
            pos_weights.append((y[:, i]==0.).sum()/y[:, i].sum())
    except:
        for i in range(y.size()[1]):
            pos_weights.append((y[:, i]==0.).sum()/y[:, i].sum())
            
    return torch.nan_to_num(torch.as_tensor(pos_weights, dtype=torch.float).to(device), posinf=0.)
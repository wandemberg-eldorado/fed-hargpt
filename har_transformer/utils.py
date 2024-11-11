from har_transformer.models import HarTransformer

def build_model(config: dict):
    input_dim = config['data_params']['input_dim']
    labels_dim = len(config['data_params']['labels']) #beware of data leak
    hidden_size = config['transformer_params']['hidden_size']
    n_positions = config['transformer_params']['n_positions']
    transformers_layers = config['transformer_params']['transformers_layers']

    model = HarTransformer(input_dim, labels_dim, hidden_size, transformers_layers=transformers_layers, n_positions=n_positions)
    model.to(config['device'])
    return model
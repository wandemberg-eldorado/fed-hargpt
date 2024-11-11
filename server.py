import flwr as fl


def main() -> None:
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        #fraction_eval=1.0,
        min_fit_clients=12,
        #min_eval_clients=12,
        min_available_clients=12,
        #eval_fn=None,
        on_fit_config_fn=fit_config,
        #on_evaluate_config_fn=evaluate_config,
        initial_parameters=None,
    )

    # Start Flower server for 10 rounds of federated learning
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=4), strategy=strategy)


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    """
    config = {
        "batch_size": 64,
        "local_epochs": 2000,
    }
    return config


def evaluate_config(rnd: int):
        """Return evaluation configuration dict for each round..
        """
        val_steps = 5 if rnd < 4 else 10
        return {"val_steps": val_steps}

if __name__ == "__main__":
    main()

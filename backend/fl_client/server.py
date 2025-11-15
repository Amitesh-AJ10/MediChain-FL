# medichain-fl/backend/fl_client/server.py

import flwr as fl
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate
import sys
from pathlib import Path

# Add backend to path to import model
sys.path.append(str(Path(__file__).parent.parent))
from model import load_model

class PartialUpdateStrategy(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters: Parameters, **kwargs):
        super().__init__(**kwargs)
        # Store the full set of initial parameters
        self.initial_parameters = initial_parameters
        
        # We need to know which layers are trainable AND the names of ALL layers.
        # Load the model ONCE to get this structural information.
        temp_model = load_model(freeze_encoder=True)
        self.trainable_param_keys = [name for name, param in temp_model.named_parameters() if param.requires_grad]
        # IMPORTANT CHANGE: Store all layer names (keys) to avoid reloading the model.
        self.all_param_keys = list(temp_model.state_dict().keys())

        print(f"Server Strategy: Identified {len(self.all_param_keys)} total layers.")
        print(f"Server Strategy: Identified {len(self.trainable_param_keys)} trainable layers: {self.trainable_param_keys}")

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """
        Initialize the global model with the full set of parameters.
        """
        print("Strategy: Initializing global model with full parameters.")
        return self.initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        
        if not results:
            print(f"🔄 ROUND {server_round} - No results from clients. Skipping aggregation.")
            # In case of failure, return the last known good model to continue
            current_full_params = self.initial_parameters if self.initial_parameters else None
            return current_full_params, {}

        print(f"\n🔄 ROUND {server_round} - Aggregating {len(results)} PARTIAL client updates.")
        
        # 1. Get the current full global model weights as a list of NumPy arrays
        current_weights = parameters_to_ndarrays(self.initial_parameters)
        
        # OPTIMIZED: Use the stored list of keys instead of reloading the model
        state_dict = dict(zip(self.all_param_keys, current_weights))

        # 2. Aggregate the partial updates from clients using FedAvg logic
        partial_updates_aggregated = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_updates = aggregate(partial_updates_aggregated)

        # 3. Surgically insert the aggregated updates back into the full model
        aggregated_updates_dict = dict(zip(self.trainable_param_keys, aggregated_updates))
        state_dict.update(aggregated_updates_dict)

        # 4. Convert the updated full state_dict back to the Flower Parameters format
        updated_full_parameters = ndarrays_to_parameters(list(state_dict.values()))
        
        # 5. Store the new full model for the next round
        self.initial_parameters = updated_full_parameters

        print(f"✅ Round {server_round} - Aggregation complete. Full model updated.")
        
        # Aggregate metrics if needed
        # We need to handle failures properly here
        aggregated_metrics = {}
        if results:
            # Re-implement a simple metric aggregation to avoid issues with the base class
            accuracies = [res.metrics.get("accuracy", 0) for _, res in results]
            if accuracies:
                aggregated_metrics["accuracy_avg"] = sum(accuracies) / len(accuracies)

        return updated_full_parameters, aggregated_metrics

def start_server(num_rounds: int = 5):
    """Starts the Flower server with our custom partial update strategy."""
    
    # The server must load the model to get the initial state
    print("Server: Loading initial model to create strategy...")
    initial_model = load_model(freeze_encoder=True)
    initial_params_list = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_params_list)

    # Instantiate our custom strategy
    strategy = PartialUpdateStrategy(
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy, 
    )

if __name__ == "__main__":
    start_server(num_rounds=5)
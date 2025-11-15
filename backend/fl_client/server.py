# medichain-fl/backend/fl_client/server.py
import flwr as fl
from typing import List, Tuple, Optional, Dict

class FedAvgStrategy(fl.server.strategy.FedAvg):
    """
    A standard FedAvg strategy for Flower, simplified without HE or Blockchain.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Flower Server initialized with standard FedAvg strategy.")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        if not results:
            print(f"🔄 ROUND {server_round} - No results from clients. Skipping aggregation.")
            return None, {}
        
        print(f"\n🔄 ROUND {server_round} - Aggregating {len(results)} client updates using FedAvg.")
        
        # Call the base FedAvg strategy's aggregate_fit method
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Extract and average accuracies for logging
        metrics_dict = {}
        if aggregated_metrics and "accuracy_avg" in aggregated_metrics:
            metrics_dict["accuracy_avg"] = aggregated_metrics["accuracy_avg"]
        else:
            # Fallback to calculate average accuracy if not already done by base class
            accuracies = [res.metrics.get("accuracy", 0) for _, res in results]
            if accuracies:
                metrics_dict["accuracy_avg"] = sum(accuracies) / len(accuracies)

        print(f"✅ Round {server_round} - Aggregation complete. Avg Accuracy: {metrics_dict.get('accuracy_avg', 0):.2f}%")
        
        return aggregated_parameters, metrics_dict


def start_server(num_rounds: int = 5):
    """Starts the Flower server with a simplified FedAvg strategy."""
    strategy = FedAvgStrategy(
        fraction_fit=1.0,         # Fraction of clients to sample for training in each round
        min_fit_clients=2,        # Minimum number of clients required for training
        min_available_clients=2,  # Minimum number of clients that need to be connected at the start
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080", # IMPORTANT CHANGE: localhost -> 0.0.0.0
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy, 
    )

if __name__ == "__main__":
    start_server(num_rounds=5)
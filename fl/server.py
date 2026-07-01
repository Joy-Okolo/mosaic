import torch
import copy
import time
from models.mosaic_model import MOSAICModel
from fl.mcat import MCATScheduler


class MOSAICServer:
    """
    Federated learning server with MCAT scheduling.
    """

    def __init__(self, config, client_profiles,
                 client_data_sizes, device='cpu'):
        self.config  = config
        self.device  = device
        self.round   = 0
        self.history = []

        # Initialize global model
        self.global_model = MOSAICModel(config).to(device)

        # Initialize MCAT scheduler
        self.scheduler = MCATScheduler(
            config, client_profiles, client_data_sizes
        )

    def get_global_state(self):
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_updates):
        """
        Weighted FedAvg aggregation.
        """
        total_samples = sum(n for _, n in client_updates)
        new_state     = copy.deepcopy(client_updates[0][0])

        for key in new_state:
            new_state[key] = torch.zeros_like(
                new_state[key], dtype=torch.float32
            )

        for state_dict, num_samples in client_updates:
            weight = num_samples / total_samples
            for key in new_state:
                new_state[key] += weight * state_dict[key].float()

        self.global_model.load_state_dict(new_state)
        return new_state

    def run_round(self, all_clients, client_indices, num_epochs):
        """
        Run one federated round using MCAT client selection.
        """
        self.round += 1
        per_round    = self.config['federation']['clients_per_round']
        global_state = self.get_global_state()

        # MCAT selects clients by priority score
        all_ids  = [c.client_id for c in all_clients]
        selected_ids = self.scheduler.select_clients(all_ids, per_round)
        selected = [c for c in all_clients if c.client_id in selected_ids]

        client_updates = []
        round_losses   = []

        print(f"  Round {self.round}: training {len(selected)} clients "
              f"(MCAT selected)...")

        for client in selected:
            start = time.time()
            updated_state, loss = client.train(global_state, num_epochs)
            elapsed = time.time() - start

            num_samples = len(client_indices[client.client_id])
            client_updates.append((updated_state, num_samples))
            round_losses.append(loss)

            # Update MCAT timing history
            self.scheduler.update_time(client.client_id, elapsed)

        self.aggregate(client_updates)

        avg_loss = sum(round_losses) / len(round_losses)
        self.history.append({'round': self.round, 'loss': avg_loss})

        return avg_loss

import torch
import copy
import time
from models.mosaic_model import MOSAICModel
from fl.mcat import MCATScheduler

class MOSAICServer:
    """
    Federated learning server with MCAT scheduling and duty-cycle availability.
    """
    def __init__(self, config, client_profiles,
                 client_data_sizes, client_tiers, device='cpu'):
        self.config       = config
        self.device       = device
        self.round        = 0
        self.history      = []
        self.client_tiers = client_tiers

        # duty-cycle: T3 clients only available every N rounds
        self.duty_cycle = config['client_tiers']['tier3_duty_cycle']

        # Initialize global model
        self.global_model = MOSAICModel(config).to(device)

        # Initialize MCAT scheduler
        self.scheduler = MCATScheduler(
            config, client_profiles, client_data_sizes
        )

    def get_global_state(self):
        return copy.deepcopy(self.global_model.state_dict())

    def _get_available_clients(self, all_client_ids):
        """
        Filter clients by duty-cycle availability.
        T1 and T2 clients: always available
        T3 clients: only available every duty_cycle rounds
        """
        available = []
        for cid in all_client_ids:
            tier = self.client_tiers.get(cid, 1)
            if tier == 3:
                # T3 clients only available when round is a multiple
                # of their duty cycle
                if self.round % self.duty_cycle == 0:
                    available.append(cid)
            else:
                available.append(cid)
        return available

    def aggregate(self, client_updates):
        """Weighted FedAvg aggregation."""
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
        Run one federated round using MCAT client selection
        with duty-cycle availability constraints.
        """
        self.round += 1
        per_round    = self.config['federation']['clients_per_round']
        global_state = self.get_global_state()

        # Filter by duty-cycle availability first
        all_ids       = [c.client_id for c in all_clients]
        available_ids = self._get_available_clients(all_ids)

        # MCAT selects from available clients only
        selected_ids = self.scheduler.select_clients(
            available_ids,
            min(per_round, len(available_ids))
        )
        selected = [c for c in all_clients if c.client_id in selected_ids]

        client_updates = []
        round_losses   = []

        for client in selected:
            start = time.time()
            updated_state, loss = client.train(global_state, num_epochs)
            elapsed = time.time() - start
            num_samples = len(client_indices[client.client_id])
            client_updates.append((updated_state, num_samples))
            round_losses.append(loss)
            self.scheduler.update_time(client.client_id, elapsed)

        self.aggregate(client_updates)
        avg_loss = sum(round_losses) / len(round_losses)
        self.history.append({'round': self.round, 'loss': avg_loss})
        return avg_loss

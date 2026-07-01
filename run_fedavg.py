import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import torch
import numpy as np
import random
import copy
from data.datasets import UCIHARDataset
from data.partition import DirichletPartitioner
from fl.client import MOSAICClient
from models.mosaic_model import MOSAICModel
from evaluation.metrics import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FedAvgServer:
    """Pure FedAvg — random client selection, simple averaging."""

    def __init__(self, config, device='cpu'):
        self.config       = config
        self.device       = device
        self.round        = 0
        self.global_model = MOSAICModel(config).to(device)

    def get_global_state(self):
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_updates):
        total   = sum(n for _, n in client_updates)
        new_state = copy.deepcopy(client_updates[0][0])
        for key in new_state:
            new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)
        for state_dict, n in client_updates:
            w = n / total
            for key in new_state:
                new_state[key] += w * state_dict[key].float()
        self.global_model.load_state_dict(new_state)

    def run_round(self, all_clients, client_indices, num_epochs):
        self.round += 1
        per_round    = self.config['federation']['clients_per_round']
        global_state = self.get_global_state()

        # RANDOM selection — no MCAT
        selected = random.sample(all_clients, per_round)

        updates      = []
        round_losses = []

        for client in selected:
            updated_state, loss = client.train(global_state, num_epochs)
            n = len(client_indices[client.client_id])
            updates.append((updated_state, n))
            round_losses.append(loss)

        self.aggregate(updates)
        return sum(round_losses) / len(round_losses)


def main():
    with open('configs/mosaic_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    set_seed(config['experiment']['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 55)
    print("FedAvg Baseline — Random Client Selection")
    print("=" * 55)
    print(f"Clients:   {config['federation']['num_clients']}")
    print(f"Rounds:    {config['federation']['num_rounds']}")
    print(f"Device:    {device}")
    print("=" * 55)

    print("\nLoading data...")
    train_dataset = UCIHARDataset(split='train')
    test_dataset  = UCIHARDataset(split='test')

    partitioner = DirichletPartitioner(
        num_clients=config['federation']['num_clients'],
        alpha=0.5,
        seed=config['experiment']['seed']
    )
    client_indices = partitioner.partition(train_dataset.get_labels())

    print("Creating clients...")
    all_clients = []
    for cid in range(config['federation']['num_clients']):
        client = MOSAICClient(
            client_id=cid,
            dataset=train_dataset,
            indices=client_indices[cid],
            config=config,
            device=device
        )
        all_clients.append(client)

    server     = FedAvgServer(config, device=device)
    num_rounds = config['federation']['num_rounds']
    log_every  = config['experiment']['log_every']
    epochs     = config['training']['local_epochs']

    acc, f1 = evaluate(server.global_model, test_dataset, device)
    print(f"\nRound   0 | Accuracy: {acc}%  F1: {f1}%")

    os.makedirs('results', exist_ok=True)
    best_acc = 0.0

    print("\nStarting FedAvg training...")
    for round_num in range(1, num_rounds + 1):
        avg_loss = server.run_round(all_clients, client_indices, epochs)

        if round_num % log_every == 0:
            acc, f1 = evaluate(server.global_model, test_dataset, device)
            print(f"Round {round_num:>3} | Loss: {avg_loss:.4f} | "
                  f"Accuracy: {acc}%  F1: {f1}%")

            if acc > best_acc:
                best_acc = acc
                torch.save(
                    server.global_model.state_dict(),
                    'results/best_fedavg_model.pt'
                )
                print(f"          New best: {best_acc}%")

    print("\n" + "=" * 55)
    print(f"FedAvg complete. Best accuracy: {best_acc}%")
    print("=" * 55)


if __name__ == '__main__':
    main()

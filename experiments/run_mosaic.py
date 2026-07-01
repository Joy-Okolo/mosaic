import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import torch
import numpy as np
import random
from data.datasets import UCIHARDataset
from data.partition import DirichletPartitioner
from data.modality_assign import ModalityAssigner
from fl.client import MOSAICClient
from fl.server import MOSAICServer
from evaluation.metrics import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lambda_anchor', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()

    with open('configs/mosaic_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    if args.lambda_anchor is not None:
        config['losses']['lambda_anchor'] = args.lambda_anchor
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr

    set_seed(config['experiment']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 55)
    print("MOSAIC Federated Learning — With MCAT Scheduling")
    print("=" * 55)
    print(f"Clients:      {config['federation']['num_clients']}")
    print(f"Rounds:       {config['federation']['num_rounds']}")
    print(f"Per round:    {config['federation']['clients_per_round']}")
    print(f"Local epochs: {config['training']['local_epochs']}")
    print(f"Lambda anchor: {config['losses']['lambda_anchor']}")
    print(f"Device:       {device}")
    print("=" * 55)

    # Load data
    print("\nLoading data...")
    train_dataset = UCIHARDataset(split='train')
    test_dataset  = UCIHARDataset(split='test')

    # Partition data
    partitioner = DirichletPartitioner(
        num_clients=config['federation']['num_clients'],
        alpha=0.5,
        seed=config['experiment']['seed']
    )
    client_indices = partitioner.partition(train_dataset.get_labels())

    # Assign modality profiles
    assigner = ModalityAssigner(
        num_clients=config['federation']['num_clients'],
        seed=config['experiment']['seed']
    )
    client_profiles, client_tiers = assigner.assign()

    # Count tiers
    t1 = sum(1 for t in client_tiers.values() if t == 1)
    t2 = sum(1 for t in client_tiers.values() if t == 2)
    t3 = sum(1 for t in client_tiers.values() if t == 3)
    print(f"Client tiers: T1={t1}  T2={t2}  T3={t3}")

    # Dataset sizes per client
    client_data_sizes = {
        cid: len(indices)
        for cid, indices in client_indices.items()
    }

    # Create all clients
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
    print(f"Created {len(all_clients)} clients")

    # Create server with MCAT
    server = MOSAICServer(
        config=config,
        client_profiles=client_profiles,
        client_data_sizes=client_data_sizes,
        device=device
    )

    num_rounds = config['federation']['num_rounds']
    log_every  = config['experiment']['log_every']
    epochs     = config['training']['local_epochs']

    # Baseline
    acc, f1 = evaluate(server.global_model, test_dataset, device)
    print(f"\nRound   0 | Accuracy: {acc}%  F1: {f1}%")

    os.makedirs('results', exist_ok=True)
    best_acc = 0.0

    print("\nStarting training with MCAT scheduling...")
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
                    'results/best_mosaic_model.pt'
                )
                print(f"          New best: {best_acc}%")

    print("\n" + "=" * 55)
    print(f"Training complete. Best accuracy: {best_acc}%")
    print("=" * 55)


if __name__ == '__main__':
    main()

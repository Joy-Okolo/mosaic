import numpy as np
from collections import defaultdict


class MCATScheduler:
    """
    Modality-Contribution-Weighted Adaptive Threshold scheduler.

    Scores each client based on three factors:
    1. Compute score  - how reliably fast the client trains
    2. Data score     - how much local data the client has
    3. Rarity score   - how rare the client modality combination is

    Clients with higher scores get priority in round selection.
    This is the core novel scheduling contribution of MOSAIC.
    """

    def __init__(self, config, client_profiles, client_data_sizes):
        self.alpha = config['mcat']['alpha']   # weight for compute
        self.beta  = config['mcat']['beta']    # weight for data
        self.gamma = config['mcat']['gamma']   # weight for rarity

        self.client_profiles   = client_profiles
        self.client_data_sizes = client_data_sizes

        # Track training times per client
        self.training_times = defaultdict(list)

        # Precompute static scores
        self.rarity_scores = self._compute_rarity()
        self.data_scores   = self._compute_data()

    def _compute_rarity(self):
        """
        Rarity = 1 / fraction of clients sharing same modality set.
        Rare modality combinations get higher scores.
        """
        from collections import Counter
        counts = Counter(
            tuple(sorted(v)) for v in self.client_profiles.values()
        )
        total = len(self.client_profiles)
        scores = {}
        for cid, profile in self.client_profiles.items():
            key = tuple(sorted(profile))
            scores[cid] = 1.0 / (counts[key] / total)

        # Normalize to [0, 1]
        max_val = max(scores.values())
        return {k: v / max_val for k, v in scores.items()}

    def _compute_data(self):
        """
        Data score = normalized local dataset size.
        More data = higher score.
        """
        total = sum(self.client_data_sizes.values())
        scores = {k: v / total for k, v in self.client_data_sizes.items()}
        max_val = max(scores.values())
        return {k: v / max_val for k, v in scores.items()}

    def update_time(self, client_id, elapsed):
        """Record how long a client took to train."""
        self.training_times[client_id].append(elapsed)
        # Keep only last 10 observations
        if len(self.training_times[client_id]) > 10:
            self.training_times[client_id].pop(0)

    def compute_score(self, client_id):
        """
        Compute MCAT priority score for one client.
        Higher = higher priority for selection.
        """
        # Compute score: inverse of average training time
        # Faster clients get higher compute score
        times = self.training_times[client_id]
        if times:
            avg_time = np.mean(times)
            # Normalize: faster = higher score
            compute_score = 1.0 / (1.0 + avg_time)
        else:
            compute_score = 0.5  # Unknown client gets neutral score

        data_score   = self.data_scores.get(client_id, 0.5)
        rarity_score = self.rarity_scores.get(client_id, 0.5)

        priority = (self.alpha * compute_score +
                    self.beta  * data_score    +
                    self.gamma * rarity_score)
        return priority

    def select_clients(self, all_client_ids, num_to_select):
        """
        Select top clients by MCAT priority score.

        Args:
            all_client_ids: list of all available client ids
            num_to_select:  how many to pick
        Returns:
            list of selected client ids
        """
        scores = {
            cid: self.compute_score(cid)
            for cid in all_client_ids
        }

        # Sort by score descending and pick top num_to_select
        selected = sorted(
            scores.keys(),
            key=lambda cid: scores[cid],
            reverse=True
        )[:num_to_select]

        return selected

import numpy as np
import os

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = "/mmfs2/home/jacks.local/joy.okolo/mosaic/data/UCI_HAR"
OUT_DIR   = "/mmfs2/home/jacks.local/joy.okolo/mosaic/baselines/harmony_ucihar_data"
NUM_CLIENTS = 100
ALPHA       = 0.5
SEED        = 42
NUM_CLASSES = 6

np.random.seed(SEED)

# ── load data ──────────────────────────────────────────────────────────────
X_train      = np.load(f"{DATA_DIR}/X_train.npy")   # (7352, 128, 6)
X_test       = np.load(f"{DATA_DIR}/X_test.npy")    # (2947, 128, 6)
y_train      = np.load(f"{DATA_DIR}/y_train.npy")   # (7352,)
y_test       = np.load(f"{DATA_DIR}/y_test.npy")    # (2947,)

# ── split into acc and gyro ────────────────────────────────────────────────
# X shape: (samples, 128, 6) → acc=[:,:,0:3], gyro=[:,:,3:6]
# Harmony Conv2d expects (batch, 1, H, W) → shape per sample: (128, 3)
acc_train  = X_train[:, :, 0:3]   # (7352, 128, 3)
gyro_train = X_train[:, :, 3:6]   # (7352, 128, 3)
acc_test   = X_test[:, :, 0:3]    # (2947, 128, 3)
gyro_test  = X_test[:, :, 3:6]    # (2947, 128, 3)

# ── Dirichlet partition on training data ───────────────────────────────────
# For each class, distribute samples across clients via Dirichlet
client_train_indices = [[] for _ in range(NUM_CLIENTS)]

for c in range(NUM_CLASSES):
    class_indices = np.where(y_train == c)[0]
    np.random.shuffle(class_indices)
    proportions = np.random.dirichlet(np.repeat(ALPHA, NUM_CLIENTS))
    proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
    splits = np.split(class_indices, proportions)
    for client_id, split in enumerate(splits):
        client_train_indices[client_id].extend(split.tolist())

# ── distribute test data equally across clients ────────────────────────────
test_indices = np.arange(len(y_test))
np.random.shuffle(test_indices)
client_test_indices = np.array_split(test_indices, NUM_CLIENTS)

# ── modality assignment ────────────────────────────────────────────────────
# T1 (both modalities): clients 0-19   → 20 clients
# T2 (acc only):        clients 20-69  → 50 clients  
# T3 (gyro only):       clients 70-99  → 30 clients
# This matches your MOSAIC tier structure

def get_modality(client_id):
    if client_id < 20:
        return 'both'
    elif client_id < 70:
        return 'acc'
    else:
        return 'gyro'

# ── save per-client data ───────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
modality_log = []

for client_id in range(NUM_CLIENTS):
    client_dir = os.path.join(OUT_DIR, f"client_{client_id}")
    os.makedirs(client_dir, exist_ok=True)

    tr_idx = client_train_indices[client_id]
    te_idx = client_test_indices[client_id]

    modality = get_modality(client_id)
    modality_log.append(f"client_{client_id}: {modality} | train={len(tr_idx)} | test={len(te_idx)}")

    if modality == 'both':
        np.save(f"{client_dir}/x_train_1.npy", acc_train[tr_idx])
        np.save(f"{client_dir}/x_train_2.npy", gyro_train[tr_idx])
        np.save(f"{client_dir}/y_train.npy",   y_train[tr_idx])
        np.save(f"{client_dir}/x_test_1.npy",  acc_test[te_idx])
        np.save(f"{client_dir}/x_test_2.npy",  gyro_test[te_idx])
        np.save(f"{client_dir}/y_test.npy",    y_test[te_idx])

    elif modality == 'acc':
        np.save(f"{client_dir}/x_train_1.npy", acc_train[tr_idx])
        np.save(f"{client_dir}/y_train.npy",   y_train[tr_idx])
        np.save(f"{client_dir}/x_test_1.npy",  acc_test[te_idx])
        np.save(f"{client_dir}/y_test.npy",    y_test[te_idx])

    else:  # gyro only
        np.save(f"{client_dir}/x_train_1.npy", gyro_train[tr_idx])
        np.save(f"{client_dir}/y_train.npy",   y_train[tr_idx])
        np.save(f"{client_dir}/x_test_1.npy",  gyro_test[te_idx])
        np.save(f"{client_dir}/y_test.npy",    y_test[te_idx])

    # save modality flag so Harmony's main script knows what this client has
    with open(f"{client_dir}/modality.txt", 'w') as f:
        f.write(modality)

# ── summary ───────────────────────────────────────────────────────────────
print(f"Done. Data saved to {OUT_DIR}")
print(f"Total clients: {NUM_CLIENTS}")
print(f"  Both modalities (T1): 20 clients (0-19)")
print(f"  Acc only       (T2): 50 clients (20-69)")
print(f"  Gyro only      (T3): 30 clients (70-99)")
print(f"\nSample client sizes (first 5):")
for line in modality_log[:5]:
    print(f"  {line}")
print(f"\nSample client sizes (last 5):")
for line in modality_log[-5:]:
    print(f"  {line}")

# save full modality log
with open(f"{OUT_DIR}/modality_log.txt", 'w') as f:
    f.write('\n'.join(modality_log))
print(f"\nFull modality log saved to {OUT_DIR}/modality_log.txt")

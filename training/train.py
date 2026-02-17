"""
ArcInfer Training Pipeline
==========================
Trains the sentiment classifier for encrypted inference on Arcium.

Steps:
  1. Load SST-2 dataset (Stanford Sentiment Treebank)
  2. Generate 384-dim embeddings using all-MiniLM-L6-v2
  3. Fit PCA (384→16) on training embeddings
  4. Train a 16→16→8→2 classifier with square activations
  5. Export weights + PCA matrix as JSON for Rust

Every step includes assertions that verify correctness.
This script IS the test — if it runs without error, the training is valid.

Usage:
  python3 training/train.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ============================================================================
# Config
# ============================================================================

EMBED_DIM = 384       # all-MiniLM-L6-v2 output dimension
PCA_DIM = 16          # target dimension after PCA (fits Solana tx limit)
HIDDEN1 = 16          # first hidden layer
HIDDEN2 = 8           # second hidden layer
NUM_CLASSES = 2       # positive / negative
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# Step 1: Load SST-2 dataset
# ============================================================================

print("Step 1: Loading SST-2 dataset...")
dataset = load_dataset("stanfordnlp/sst2")
train_texts = dataset["train"]["sentence"]
train_labels = dataset["train"]["label"]
val_texts = dataset["validation"]["sentence"]
val_labels = dataset["validation"]["label"]

print(f"  Train: {len(train_texts)} samples")
print(f"  Val:   {len(val_texts)} samples")

# VERIFY: SST-2 should have ~67K train, ~872 validation
assert len(train_texts) > 60000, f"Expected >60K train samples, got {len(train_texts)}"
assert len(val_texts) > 800, f"Expected >800 val samples, got {len(val_texts)}"
# VERIFY: Labels are binary (0 or 1)
assert set(train_labels) == {0, 1}, f"Expected binary labels, got {set(train_labels)}"


# ============================================================================
# Step 2: Generate embeddings
# ============================================================================

print("\nStep 2: Generating embeddings with all-MiniLM-L6-v2...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings in batches
train_embeddings = embedder.encode(train_texts, show_progress_bar=True, batch_size=256)
val_embeddings = embedder.encode(val_texts, show_progress_bar=True, batch_size=256)

train_embeddings = np.array(train_embeddings, dtype=np.float64)
val_embeddings = np.array(val_embeddings, dtype=np.float64)

print(f"  Train embeddings shape: {train_embeddings.shape}")
print(f"  Val embeddings shape:   {val_embeddings.shape}")

# VERIFY: Correct dimensions
assert train_embeddings.shape == (len(train_texts), EMBED_DIM), \
    f"Expected ({len(train_texts)}, {EMBED_DIM}), got {train_embeddings.shape}"
# VERIFY: No NaN or Inf values
assert np.all(np.isfinite(train_embeddings)), "Train embeddings contain NaN/Inf"
assert np.all(np.isfinite(val_embeddings)), "Val embeddings contain NaN/Inf"
# VERIFY: Embeddings are in reasonable range
assert np.abs(train_embeddings).max() < 10.0, \
    f"Embedding values unexpectedly large: max={np.abs(train_embeddings).max()}"


# ============================================================================
# Step 3: Fit PCA (384 → 16)
# ============================================================================

print(f"\nStep 3: Fitting PCA ({EMBED_DIM} → {PCA_DIM})...")
pca = PCA(n_components=PCA_DIM, random_state=SEED)
train_pca = pca.fit_transform(train_embeddings)
val_pca = pca.transform(val_embeddings)

explained_variance = pca.explained_variance_ratio_.sum()
print(f"  Explained variance: {explained_variance:.4f} ({explained_variance*100:.1f}%)")
print(f"  Train PCA shape: {train_pca.shape}")
print(f"  Val PCA shape:   {val_pca.shape}")

# VERIFY: Correct output dimensions
assert train_pca.shape == (len(train_texts), PCA_DIM), \
    f"Expected ({len(train_texts)}, {PCA_DIM}), got {train_pca.shape}"
# VERIFY: Reasonable variance retained
# Note: sentence-transformer embeddings spread information across many dimensions.
# 16 components from 384 captures ~30% variance, which is sufficient for binary
# classification — the classifier learns to use whatever PCA preserves.
assert explained_variance > 0.20, \
    f"PCA retains only {explained_variance:.1%} variance — too low"
# VERIFY: PCA components have correct shape
assert pca.components_.shape == (PCA_DIM, EMBED_DIM), \
    f"Expected components ({PCA_DIM}, {EMBED_DIM}), got {pca.components_.shape}"
# VERIFY: Mean vector has correct shape
assert pca.mean_.shape == (EMBED_DIM,), \
    f"Expected mean ({EMBED_DIM},), got {pca.mean_.shape}"


# ============================================================================
# Step 4: Train the classifier
# ============================================================================

print(f"\nStep 4: Training classifier ({PCA_DIM}→{HIDDEN1}→{HIDDEN2}→{NUM_CLASSES})...")


class SquareActivation(nn.Module):
    """f(x) = x². The MPC-friendly activation function.

    In MPC, this costs 2 rounds (one multiplication via Beaver triple).
    ReLU would cost 20-40 rounds (requires bit decomposition for comparison).
    """
    def forward(self, x):
        return x * x


class SentimentClassifier(nn.Module):
    """The exact architecture that will run inside the Arcis circuit."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PCA_DIM, HIDDEN1),
            SquareActivation(),
            nn.Linear(HIDDEN1, HIDDEN2),
            SquareActivation(),
            nn.Linear(HIDDEN2, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


model = SentimentClassifier()

# VERIFY: Parameter count matches our architecture doc
total_params = sum(p.numel() for p in model.parameters())
expected_params = (PCA_DIM * HIDDEN1 + HIDDEN1) + (HIDDEN1 * HIDDEN2 + HIDDEN2) + (HIDDEN2 * NUM_CLASSES + NUM_CLASSES)
print(f"  Total parameters: {total_params}")
assert total_params == expected_params, \
    f"Expected {expected_params} params, got {total_params}"

# Prepare data
train_X = torch.tensor(train_pca, dtype=torch.float32)
train_y = torch.tensor(train_labels, dtype=torch.long)
val_X = torch.tensor(val_pca, dtype=torch.float32)
val_y = torch.tensor(val_labels, dtype=torch.long)

train_dataset = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training loop
best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_X.size(0)

    train_acc = correct / total
    train_loss = total_loss / total

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(val_X)
        val_preds = val_logits.argmax(dim=1)
        val_acc = (val_preds == val_y).float().mean().item()

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}: loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

print(f"\n  Best validation accuracy: {best_val_acc:.4f}")

# VERIFY: Model should achieve reasonable accuracy
# Square activations are weaker than ReLU, but should still get >75% on SST-2
assert best_val_acc > 0.75, \
    f"Validation accuracy too low: {best_val_acc:.4f}. Expected >75%"
print(f"  ✓ Accuracy check passed (>{75}%)")


# ============================================================================
# Step 5: Verify weights are in Q16.16 safe range
# ============================================================================

print("\nStep 5: Checking weight ranges for Q16.16 compatibility...")
MAX_Q16_16 = 32767.0
all_weights_safe = True
for name, param in model.named_parameters():
    max_val = param.abs().max().item()
    print(f"  {name}: max |value| = {max_val:.6f}")
    if max_val > MAX_Q16_16:
        print(f"    ⚠ WARNING: exceeds Q16.16 range!")
        all_weights_safe = False

assert all_weights_safe, "Some weights exceed Q16.16 range — need weight clipping or rescaling"
print("  ✓ All weights within Q16.16 range")


# ============================================================================
# Step 6: Export everything as JSON
# ============================================================================

print(f"\nStep 6: Exporting to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Export PCA
pca_export = {
    "input_dim": EMBED_DIM,
    "output_dim": PCA_DIM,
    "mean": pca.mean_.tolist(),
    "components": pca.components_.tolist(),
    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
}

pca_path = os.path.join(OUTPUT_DIR, "pca.json")
with open(pca_path, "w") as f:
    json.dump(pca_export, f)
print(f"  Saved PCA to {pca_path}")

# VERIFY: PCA JSON roundtrips correctly
with open(pca_path) as f:
    pca_check = json.load(f)
assert len(pca_check["mean"]) == EMBED_DIM
assert len(pca_check["components"]) == PCA_DIM
assert len(pca_check["components"][0]) == EMBED_DIM

# Export classifier weights
weights_export = {}
for name, param in model.named_parameters():
    weights_export[name] = param.detach().cpu().numpy().tolist()

weights_path = os.path.join(OUTPUT_DIR, "classifier_weights.json")
with open(weights_path, "w") as f:
    json.dump(weights_export, f)
print(f"  Saved classifier weights to {weights_path}")

# VERIFY: Weight JSON roundtrips correctly
with open(weights_path) as f:
    w_check = json.load(f)
# Layer 1: 16→16
assert len(w_check["net.0.weight"]) == HIDDEN1
assert len(w_check["net.0.weight"][0]) == PCA_DIM
assert len(w_check["net.0.bias"]) == HIDDEN1
# Layer 2: 16→8
assert len(w_check["net.2.weight"]) == HIDDEN2
assert len(w_check["net.2.weight"][0]) == HIDDEN1
assert len(w_check["net.2.bias"]) == HIDDEN2
# Layer 3: 8→2
assert len(w_check["net.4.weight"]) == NUM_CLASSES
assert len(w_check["net.4.weight"][0]) == HIDDEN2
assert len(w_check["net.4.bias"]) == NUM_CLASSES

# Export metadata
metadata = {
    "architecture": f"{PCA_DIM}→{HIDDEN1}→{HIDDEN2}→{NUM_CLASSES}",
    "activation": "square (x²)",
    "pca_explained_variance": float(explained_variance),
    "best_val_accuracy": float(best_val_acc),
    "total_parameters": total_params,
    "train_samples": len(train_texts),
    "val_samples": len(val_texts),
    "epochs": EPOCHS,
    "seed": SEED,
}

metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata to {metadata_path}")


# ============================================================================
# Final verification: end-to-end inference on known examples
# ============================================================================

print("\nFinal verification: end-to-end inference...")

test_sentences = [
    ("This movie is absolutely wonderful and I loved every moment of it.", 1),
    ("Terrible film. Waste of time and money.", 0),
    ("A masterpiece of modern cinema.", 1),
    ("Boring, predictable, and poorly acted.", 0),
]

model.eval()
correct = 0
for sentence, expected_label in test_sentences:
    # Full pipeline: text → embed → PCA → classify
    embedding = embedder.encode([sentence], show_progress_bar=False)
    embedding = np.array(embedding, dtype=np.float64)
    pca_input = pca.transform(embedding)
    tensor_input = torch.tensor(pca_input, dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor_input)
        predicted = logits.argmax(dim=1).item()

    label_str = "positive" if predicted == 1 else "negative"
    status = "✓" if predicted == expected_label else "✗"
    print(f"  {status} \"{sentence[:50]}...\" → {label_str} (expected {expected_label})")
    if predicted == expected_label:
        correct += 1

print(f"\n  End-to-end: {correct}/{len(test_sentences)} correct")

# We don't assert 100% here — some edge cases may be wrong with square activations.
# But we do want at least 3/4 correct as a sanity check.
assert correct >= 3, f"Only {correct}/{len(test_sentences)} correct — model may be broken"

print("\n" + "="*60)
print(f"Training complete!")
print(f"  {total_params} parameters")
print(f"  {best_val_acc:.1%} validation accuracy")
print(f"  {explained_variance:.1%} PCA variance retained")
print(f"  Files saved to {OUTPUT_DIR}/")
print("="*60)

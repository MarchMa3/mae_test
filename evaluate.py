# compare_checkpoints.py
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from MAE_v2 import mae_small
from utils_v2 import LabDataset, collate_fn_fixed_length


def load_test_data(data_path):
    """Load test set (last 20% of data)"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    vocab_size = int(data['vocab_info']['vocab_size'])
    patient_sequences = data['patient_sequences']
    
    all_loinc, all_value, all_masks = [], [], []
    for seq_data in patient_sequences.values():
        all_loinc.append(seq_data['loinc_tokens'])
        all_value.append(seq_data['value_tokens'])
        all_masks.append(seq_data['missing_mask'])
    
    max_len = max(len(x) for x in all_loinc)
    n_samples = len(all_loinc)
    
    padded_loinc = np.zeros((n_samples, max_len), dtype=np.int64)
    padded_value = np.zeros((n_samples, max_len), dtype=np.int64)
    padded_masks = np.zeros((n_samples, max_len), dtype=np.float32)
    
    for i, (loinc, value, mask) in enumerate(zip(all_loinc, all_value, all_masks)):
        L = len(loinc)
        padded_loinc[i, :L] = loinc
        padded_value[i, :L] = value
        padded_masks[i, :L] = mask
    
    n_train = int(0.8 * n_samples)
    return padded_loinc[n_train:], padded_value[n_train:], padded_masks[n_train:], vocab_size


@torch.no_grad()
def evaluate_checkpoint(model, test_loader, device, mask_ratio=0.5):
    """Evaluate accuracy on test set"""
    model.eval()
    
    # Get reference embeddings for bins
    value_vocab = torch.arange(4, 14, device=device)  # bins 0-9 -> tokens 4-13
    ref_embeds = model.value_embedding(value_vocab)
    ref_embeds = ref_embeds / (ref_embeds.norm(dim=-1, keepdim=True) + 1e-8)
    
    total_correct = 0
    total_count = 0
    
    for batch in tqdm(test_loader, desc="Evaluating", ncols=80):
        loinc = batch['loinc_tokens'].to(device, non_blocking=True)
        value = batch['value_tokens'].to(device, non_blocking=True)
        missing = batch['missing_mask'].to(device, non_blocking=True)
        lengths = batch['actual_lengths'].to(device, non_blocking=True)
        
        with autocast():
            pred_embeds, mask = model.reconstruct(loinc, value, missing, mask_ratio, lengths)
        
        pred_embeds = pred_embeds.float()
        valid = (mask == 1) & (missing == 1) & (loinc != 1)  # masked & not missing & not CLS
        
        if valid.sum() == 0:
            continue
        
        pred_valid = pred_embeds[valid]
        true_valid = value[valid]
        
        # Find nearest bin
        pred_valid = pred_valid / (pred_valid.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = pred_valid @ ref_embeds.T
        pred_bins = similarity.argmax(dim=-1)
        true_bins = true_valid - 4
        
        total_correct += (pred_bins == true_bins).sum().item()
        total_count += len(pred_bins)
    
    return 100.0 * total_correct / max(total_count, 1)


def main():
    # Config
    data_path = 'data/labevents_mimic.pkl'
    checkpoint_epochs = [10, 20, 30, 40, 50]
    
    fixed_length = 120
    input_dim = 64
    mask_ratio = 0.5
    batch_size = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    print("Loading test data...")
    test_loinc, test_value, test_masks, vocab_size = load_test_data(data_path)
    test_dataset = LabDataset(test_loinc, test_value, test_masks)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
        collate_fn=lambda b: collate_fn_fixed_length(b, fixed_length),
    )
    print(f"Test set: {len(test_loinc):,} sequences\n")
    
    # Evaluate each checkpoint
    results = []
    for epoch in checkpoint_epochs:
        ckpt_path = f'checkpoints_mimic/epoch_{epoch}.pth'
        print(f"\nEpoch {epoch}:")
        
        try:
            # Load model
            model = mae_small(
                seq_len=fixed_length, vocab_size=vocab_size,
                num_bins=10, input_dim=input_dim,
                use_cls_token=True, mask_ratio=mask_ratio,
                exclude_columns=[0]
            )
            
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            
            # Remove DDP prefix if exists
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            model = model.to(device)
            
            # Evaluate
            accuracy = evaluate_checkpoint(model, test_loader, device, mask_ratio)
            results.append({'epoch': epoch, 'accuracy': accuracy})
            print(f"Accuracy: {accuracy:.2f}%")
            
            del model, checkpoint
            torch.cuda.empty_cache()
            
        except FileNotFoundError:
            print(f"Checkpoint not found: {ckpt_path}")
    
    # Plot
    if results:
        epochs = [r['epoch'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accuracies, 'o-', linewidth=2, markersize=8, color='steelblue')
        plt.axhline(y=10, color='red', linestyle='--', label='Random (10%)')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.title('Test Accuracy vs Training Epoch', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('test_accuracy_curve.png', dpi=300)
        print(f"\n✓ Plot saved: test_accuracy_curve.png")


if __name__ == "__main__":
    main()
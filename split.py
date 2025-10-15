# split_data.py
import pickle
import numpy as np
import os


def split_test_set(
    input_pkl='data/labevents_mimic.pkl',
    output_dir='data',
    test_ratio=0.2,
    seed=42
):
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)
    
    patient_sequences = data['patient_sequences']
    vocab_info = data['vocab_info']
    all_keys = list(patient_sequences.keys())
    
    np.random.seed(seed)
    indices = np.random.permutation(len(all_keys))
    
    n_test = int(test_ratio * len(all_keys))
    test_idx = indices[:n_test]
    trainval_idx = indices[n_test:]
    
    trainval_sequences = {all_keys[i]: patient_sequences[all_keys[i]] for i in trainval_idx}
    test_sequences = {all_keys[i]: patient_sequences[all_keys[i]] for i in test_idx}
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/labevents_mimic_trainval.pkl', 'wb') as f:
        pickle.dump({
            'patient_sequences': trainval_sequences,
            'vocab_info': vocab_info
        }, f, protocol=4)
    
    with open(f'{output_dir}/labevents_mimic_test.pkl', 'wb') as f:
        pickle.dump({
            'patient_sequences': test_sequences,
            'vocab_info': vocab_info
        }, f, protocol=4)
    
    print(f"Train/Val set: {len(trainval_sequences)} patients")
    print(f"Test set: {len(test_sequences)} patients")


if __name__ == "__main__":
    split_test_set()
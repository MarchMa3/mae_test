import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import json
import h5py

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_vocabulary(df):
    """
    Create a vocabulary from a dataframe of sentences.
    """

    SPECIAL_TOKENS = {
        'PAD': 0,
        'CLS': 1,
        'MASK': 2,
        'MISSING': 3,
    }

    # Loinc Code
    unique_loinc_code = sorted(df['loinc_code'].unique())
    lab_start_id = len(SPECIAL_TOKENS)
    loinc_code2token = {loinc: idx + lab_start_id for idx, loinc in enumerate(unique_loinc_code)}

    all_token_mappings = {
        **SPECIAL_TOKENS,
        **{f"LOINC_{loinc}": token_id for loinc, token_id in loinc_code2token.items()},
    }

    vocab_info = {
        'special_tokens': SPECIAL_TOKENS,
        'loinc_to_token_id': loinc_code2token,
        'token_id_to_loinc': {v: k for k, v in loinc_code2token.items()},
        'vocab_size': len(SPECIAL_TOKENS) + len(unique_loinc_code),
        'lab_event_count': len(unique_loinc_code),
        'lab_token_range': (lab_start_id, lab_start_id + len(unique_loinc_code)),
    }
    return vocab_info

def build_sequent(df, vocab_info):
    """
    Create token sequence for each patient id
    Format: [CLS, test1, test2, ..]
    Deal with loinc code and lab test value seperately.
    """
    CLS_TOKEN = vocab_info['special_tokens']['CLS']
    MISSING_TOKEN = vocab_info['special_tokens']['MISSING']
    loinc2tokenid = vocab_info['loinc_to_token_id']

    df['date'] = pd.to_datetime(df['charttime']).dt.date
    patient_daily_seq = {}
    missing_value_count = 0
    total_records = 0
    grouped = df.groupby(['patient_id', 'date'])

    for (patient_id, date), day_data in grouped:
        loinc_tokens = [CLS_TOKEN]
        value_tokens = [0.0]
        
        day_data = day_data.sort_values('charttime', kind='stable')
        for _, row in day_data.iterrows():
            loinc_code = row['loinc_code']
            value = row['value']
            
            total_records += 1
            
            # Lab token
            lab_token = loinc2tokenid[loinc_code]
            loinc_tokens.append(lab_token)
            
            # Value 
            if pd.isna(value):
                value_tokens.append(float(MISSING_TOKEN))
                missing_value_count += 1
            else:
                value_tokens.append(float(value))

        seq_key = (patient_id, str(date))
        patient_daily_seq[seq_key] = {
            'loinc_tokens': loinc_tokens,
            'value_tokens': value_tokens,
            'sequence_length': len(loinc_tokens),
            'lab_event_count': len(loinc_tokens) - 1, 
        }
    
    # Print statistics
    sequence_lengths = [seq['sequence_length'] for seq in patient_daily_seq.values()]
    lab_counts = [seq['lab_event_count'] for seq in patient_daily_seq.values()]
    
    print(f"\n=== Sequence Building Complete ===")
    print(f"Total sequences created: {len(patient_daily_seq)}")
    print(f"Sequence format: [CLS, test1, test2, test3, ...]")
    print(f"  - loinc_tokens: [CLS, loinc1, loinc2, ...]")
    print(f"  - value_tokens: [0.0, value1, value2, ...]")
    print(f"Missing values represented by MISSING token ({MISSING_TOKEN}) in value_tokens")
    print(f"\nMissing information:")
    print(f"  Missing lab values: {missing_value_count}/{total_records} ({missing_value_count/total_records*100:.1f}%)")
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(sequence_lengths)}")
    print(f"  Max: {max(sequence_lengths)}")
    print(f"\nLab events per day statistics:")
    print(f"  Min: {min(lab_counts)}")
    print(f"  Max: {max(lab_counts)}")

    return patient_daily_seq


def main():
    """
    Main function to build and display token sequences
    """
    try:
        df = pd.read_csv('data/clean/norm/normalized_data.csv')
        
        # Check for required columns
        required_cols = ['patient_id', 'charttime', 'loinc_code', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return

        # Create vocabulary
        vocab_info = create_vocabulary(df)
        MISSING_TOKEN = vocab_info['special_tokens']['MISSING']

        # Build sequences
        patient_sequences = build_sequent(df, vocab_info)

        # Display token sequences WITH MASKS
        print("\n" + "="*60)
        print("TOKEN SEQUENCES WITH MASKS")
        print("="*60)
        
        for i, (seq_key, seq_data) in enumerate(list(patient_sequences.items())[:5]): 
            patient_id, date = seq_key
            loinc_tokens = seq_data['loinc_tokens']
            value_tokens = seq_data['value_tokens']
            seq_length = seq_data['sequence_length']
            
            # Create missing_mask: 1=observed value, 0=missing value
            missing_mask = np.array([0.0 if val == MISSING_TOKEN else 1.0 
                                     for val in value_tokens], dtype=np.float32)
            
            print(f"\n{i+1}. Patient: {patient_id}, Date: {date}")
            print(f"   loinc_tokens: {loinc_tokens}")
            print(f"   value_tokens: {value_tokens}")
            print(f"   missing_mask: {missing_mask.tolist()}")
            print(f"   length: {seq_length}, labs: {seq_data['lab_event_count']}")
            print(f"   missing count: {(missing_mask == 0).sum()}/{seq_length}")

        # Save outputs
        with open('data/vocabulary_v2.json', 'w') as f:
            json.dump(vocab_info, f, indent=2, default=str)
    
        print(f"\n✓ Vocabulary saved to 'vocabulary_v2.json'")

        # Save sequences as HDF5
        output_h5_path = 'data/labevents_v2.h5'
        with h5py.File(output_h5_path, 'w') as hf:
            # Create a group for sequences
            seq_group = hf.create_group('sequences')
            
            for seq_key, seq_data in patient_sequences.items():
                patient_id, date = seq_key
                # Create unique key for HDF5 (replace special characters)
                h5_key = f"{patient_id}_{date}".replace('-', '_').replace(':', '_')
                
                # Create subgroup for each patient-date combination
                patient_group = seq_group.create_group(h5_key)
                
                # Convert to numpy arrays
                loinc_tokens_array = np.array(seq_data['loinc_tokens'], dtype=np.int64)
                value_tokens_array = np.array(seq_data['value_tokens'], dtype=np.float64)
                seq_length = seq_data['sequence_length']
                
                # Create missing_mask: 1=observed, 0=missing
                missing_mask = np.array([0.0 if val == MISSING_TOKEN else 1.0 
                                        for val in value_tokens_array], dtype=np.float32)
                
                # Store data 
                patient_group.create_dataset('loinc_tokens', data=loinc_tokens_array)
                patient_group.create_dataset('value_tokens', data=value_tokens_array)
                patient_group.create_dataset('missing_mask', data=missing_mask)
                patient_group.create_dataset('sequence_length', data=seq_length)
                patient_group.create_dataset('lab_event_count', data=seq_data['lab_event_count'])
                
                # Store metadata as attributes
                patient_group.attrs['patient_id'] = str(patient_id)
                patient_group.attrs['date'] = str(date)
            
            # Store vocabulary info as attributes in root
            hf.attrs['vocab_size'] = vocab_info['vocab_size']
            hf.attrs['lab_event_count'] = vocab_info['lab_event_count']
            hf.attrs['total_sequences'] = len(patient_sequences)
            
            # Store special tokens
            special_tokens_group = hf.create_group('special_tokens')
            for token_name, token_id in vocab_info['special_tokens'].items():
                special_tokens_group.attrs[token_name] = token_id
            
            # Store LOINC mappings
            loinc_group = hf.create_group('loinc_mappings')
            for loinc, token_id in vocab_info['loinc_to_token_id'].items():
                loinc_group.attrs[str(loinc)] = token_id
        
        print(f"✓ Sequences saved to '{output_h5_path}'")
        
        # Verify the saved file with detailed mask info
        print(f"\n=== HDF5 File Structure ===")
        with h5py.File(output_h5_path, 'r') as hf:
            print(f"Root attributes: {dict(hf.attrs)}")
            print(f"Groups: {list(hf.keys())}")
            print(f"Total sequences stored: {len(hf['sequences'].keys())}")
            
            # Check first sequence structure
            first_seq_key = list(hf['sequences'].keys())[0]
            first_seq = hf['sequences'][first_seq_key]
            print(f"\nFirst sequence datasets:")
            for key in first_seq.keys():
                dataset = first_seq[key]
                print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                if key == 'missing_mask':
                    mask_data = dataset[:]
                    print(f"    Values: 1={np.sum(mask_data==1)}, 0={np.sum(mask_data==0)}")
                    print(f"    First 30: {mask_data[:30].tolist()}")
                elif key in ['loinc_tokens', 'value_tokens']:
                    print(f"    First 10: {dataset[:10].tolist()}")

        return vocab_info, patient_sequences

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
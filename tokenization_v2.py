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
        'loinc_to_token_id': {str(k): int(v) for k, v in loinc_code2token.items()},  
        'token_id_to_loinc': {int(v): str(k) for k, v in loinc_code2token.items()}, 
        'vocab_size': len(SPECIAL_TOKENS) + len(unique_loinc_code),
        'lab_event_count': len(unique_loinc_code),
        'lab_token_range': (lab_start_id, lab_start_id + len(unique_loinc_code)),
    }
    return vocab_info

def build_sequent(df, vocab_info):
    """
    Create token sequence for each patient id - OPTIMIZED VERSION
    Format: [CLS, test1, test2, ..]
    Deal with loinc code and lab test value separately.
    """
    CLS_TOKEN = vocab_info['special_tokens']['CLS']
    MISSING_TOKEN = vocab_info['special_tokens']['MISSING']
    loinc2tokenid = vocab_info['loinc_to_token_id']

    # Pre-convert date column once
    df['date'] = pd.to_datetime(df['charttime']).dt.date
    
    patient_daily_seq = {}
    missing_value_count = 0
    total_records = 0
    
    # Sort once before grouping
    df_sorted = df.sort_values(['patient_id', 'date', 'charttime'])
    grouped = df_sorted.groupby(['patient_id', 'date'])

    for (patient_id, date), day_data in grouped:
        # Use vectorized operations instead of iterrows
        loinc_codes = day_data['loinc_code'].values
        values = day_data['value'].values
        
        # Create tokens using vectorized operations
        loinc_tokens = [CLS_TOKEN] + [loinc2tokenid[str(lc)] for lc in loinc_codes]
        
        # Handle missing values vectorized
        value_tokens = [0.0]
        for val in values:
            total_records += 1
            if pd.isna(val):
                value_tokens.append(float(MISSING_TOKEN))
                missing_value_count += 1
            else:
                value_tokens.append(float(val))

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

    return patient_daily_seq, {
        'missing_value_count': missing_value_count,
        'total_records': total_records
    }


def main():
    """
    Main function to build and display token sequences
    """
    try:
        print("Loading data...")
        df = pd.read_csv('data/clean/norm/normalized_data.csv')
        
        # Check for required columns
        required_cols = ['patient_id', 'charttime', 'loinc_code', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return

        # Create vocabulary
        print("Creating vocabulary...")
        vocab_info = create_vocabulary(df)
        MISSING_TOKEN = vocab_info['special_tokens']['MISSING']

        # Build sequences
        print("Building sequences...")
        patient_sequences, build_stats = build_sequent(df, vocab_info)  

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
        print("\nSaving vocabulary...")
        with open('data/vocabulary_mimic.json', 'w') as f:
            json.dump(vocab_info, f, indent=2, default=str)
    
        print(f"✓ Vocabulary saved to 'vocabulary_v2.json'")

        # Save sequences as pkl
        print("Saving sequences to Pickle...")
        import pickle

        output_data = {
            'patient_sequences': patient_sequences,
            'vocab_info': vocab_info,
            'MISSING_TOKEN': MISSING_TOKEN,
            'metadata': {
                'total_sequences': len(patient_sequences),
                'vocab_size': vocab_info['vocab_size'],
                'lab_event_count': vocab_info['lab_event_count'],
                'missing_value_count': build_stats['missing_value_count'],
                'total_records': build_stats['total_records']  
            }
        }

        output_pkl_path = 'data/labevents_mimic.pkl'
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=4)

        print(f"✓ Sequences saved to '{output_pkl_path}'")

        # Quick verification
        print(f"\n=== Pickle File Info ===")
        print(f"Total sequences: {len(patient_sequences):,}")
        print(f"Vocab size: {vocab_info['vocab_size']}")
        print(f"File saved at: {output_pkl_path}")

        # Show how to load
        print(f"\nTo load this data:")
        print(f"  import pickle")
        print(f"  with open('{output_pkl_path}', 'rb') as f:")
        print(f"      data = pickle.load(f)")
        print(f"  patient_sequences = data['patient_sequences']")
        print(f"  vocab_info = data['vocab_info']")

        return vocab_info, patient_sequences
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
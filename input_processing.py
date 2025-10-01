import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import json

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

    # Unit
    unique_units = sorted(df['unit'].dropna().unique())
    unit_start_id = lab_start_id + len(unique_loinc_code)
    unit2token = {unit: idx + unit_start_id for idx, unit in enumerate(unique_units)}

    all_token_mappings = {
        **SPECIAL_TOKENS,
        **{f"LOINC_{loinc}": token_id for loinc, token_id in loinc_code2token.items()},
        **{f"UNIT_{unit}": token_id for unit, token_id in unit2token.items()}
    }

    vocab_info = {
        'special_tokens': SPECIAL_TOKENS,
        'loinc_to_token_id': loinc_code2token,
        'unit_to_token_id': unit2token,
        'token_id_to_loinc': {v: k for k, v in loinc_code2token.items()},
        'token_id_to_unit': {v: k for k, v in unit2token.items()},
        'vocab_size': len(SPECIAL_TOKENS) + len(unique_loinc_code) + len(unique_units),
        'lab_event_count': len(unique_loinc_code),
        'unit_count': len(unique_units),
        'lab_token_range': (lab_start_id, lab_start_id + len(unique_loinc_code)),
        'unit_token_range': (unit_start_id, unit_start_id + len(unique_units))
    }
    return vocab_info

def build_sequent(df, vocab_info):
    """
    Create token sequence for each patient id
    """
    CLS_TOKEN = vocab_info['special_tokens']['CLS']
    PAD_TOKEN = vocab_info['special_tokens']['PAD']
    loinc2tokenid = vocab_info['loinc_to_token_id']
    unit2tokenid = vocab_info['unit_to_token_id']

    df['date'] = pd.to_datetime(df['charttime']).dt.date
    patient_daily_seq = {}
    missing_value_count = 0
    nan_unit_count = 0
    total_records = 0
    grouped = df.groupby(['patient_id', 'date'])

    for (patient_id, date), day_data in grouped:
        tokens = [CLS_TOKEN]
        values = [1.0]
        units = []

        day_data = day_data.sort_values('charttime', kind='stable')
        for _, row in day_data.iterrows():
            loinc_code = row['loinc_code']
            value = row['value']
            unit = row['unit']
            
            total_records += 1
            
            # Lab token should always be the loinc code token (not PAD)
            lab_token = loinc2tokenid[loinc_code]
            tokens.append(lab_token)
            
            # Handle missing values - only the value is 0.0, not the token
            if pd.isna(value):
                values.append(0.0)
                missing_value_count += 1
            else:
                values.append(float(value))

            # Handle unit token
            if pd.isna(unit):
                unit_token = PAD_TOKEN
                units.append(None)
                nan_unit_count += 1
            else:
                unit_token = unit2tokenid[unit]
                units.append(unit)

            tokens.append(unit_token)

        seq_key = (patient_id, str(date))
        # Store sequence data (without patient_id and date as separate fields)
        patient_daily_seq[seq_key] = {
            'tokens': tokens,
            'values': values,
            'units': units,
            'sequence_length': len(tokens),
            'lab_event_count': (len(tokens) - 1) // 2,
        }
    
    # Print statistics
    sequence_lengths = [seq['sequence_length'] for seq in patient_daily_seq.values()]
    lab_counts = [seq['lab_event_count'] for seq in patient_daily_seq.values()]
    
    print(f"\n=== Sequence Building Complete ===")
    print(f"Total sequences created: {len(patient_daily_seq)}")
    print(f"Sequence format: [CLS, lab1, unit1, lab2, unit2, ...]")
    print(f"Missing values represented by 0.0 in values array")
    print(f"\nMissing information:")
    print(f"  Missing lab values (0.0 in values): {missing_value_count}/{total_records} ({missing_value_count/total_records*100:.1f}%)")
    print(f"  Missing units (PAD token): {nan_unit_count}")
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(sequence_lengths)}")
    print(f"  Max: {max(sequence_lengths)}")
    print(f"  Mean: {np.mean(sequence_lengths):.1f}")
    print(f"  Median: {np.median(sequence_lengths):.1f}")
    print(f"\nLab events per day statistics:")
    print(f"  Min: {min(lab_counts)}")
    print(f"  Max: {max(lab_counts)}")
    print(f"  Mean: {np.mean(lab_counts):.1f}")
    print(f"  Median: {np.median(lab_counts):.1f}")

    return patient_daily_seq



def main():
    """
    Main function to build and display token sequences
    """
    try:
        df = pd.read_csv('data/clean/norm/normalized_data.csv')
        
        # Check for required columns
        required_cols = ['patient_id', 'charttime', 'loinc_code', 'value', 'unit']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return

        # Create vocabulary
        vocab_info = create_vocabulary(df)

        # Build sequences
        patient_sequences = build_sequent(df, vocab_info)

        # Display token sequences
        print("\n" + "="*60)
        print("TOKEN SEQUENCES")
        print("="*60)
        
        for i, (seq_key, seq_data) in enumerate(list(patient_sequences.items())):
            patient_id, date = seq_key
            print(f"\n {i+1} Patient: {patient_id}, Date: {date}")
            print(f"tokens:  {seq_data['tokens']}")
            print(f"values:  {seq_data['values']}")
            print(f"units:   {seq_data['units']}")

        # Save outputs
        with open('vocabulary.json', 'w') as f:
            json.dump(vocab_info, f, indent=2, default=str)
        
        with open('patient_sequences.pkl', 'wb') as f:
            pickle.dump(patient_sequences, f)

        return vocab_info, patient_sequences

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
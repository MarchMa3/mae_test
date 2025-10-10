import pandas as pd
import numpy as np
import pickle
import os
import json


def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_vocabulary(df, num_bins: int = 10):
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
    unique_loinc_code = sorted(df['loinc_code'].astype(str).unique())
    lab_start_id = len(SPECIAL_TOKENS)
    loinc_code2token = {loinc: idx + lab_start_id for idx, loinc in enumerate(unique_loinc_code)}

    vocab_info = {
        'special_tokens': SPECIAL_TOKENS,
        'loinc_to_token_id': {str(k): int(v) for k, v in loinc_code2token.items()},  
        'token_id_to_loinc': {int(v): str(k) for k, v in loinc_code2token.items()}, 
        'vocab_size': len(SPECIAL_TOKENS) + len(unique_loinc_code),
        'lab_event_count': len(unique_loinc_code),
        'lab_token_range': (lab_start_id, lab_start_id + len(unique_loinc_code) - 1),
        'num_bins': int(num_bins),
        'pad_id': 0, 
    }
    return vocab_info

def build_sequence(df, vocab_info):
    """
    Create token sequence for each patient id - OPTIMIZED VERSION
    Format: [CLS, test1, test2, ..]
    Deal with loinc code and lab test value separately.
    """
    SPECIAL_TOKEN = vocab_info['special_tokens']
    PAD_TOKEN = SPECIAL_TOKEN['PAD']
    CLS_TOKEN = SPECIAL_TOKEN['CLS']
    MISSING_TOKEN = SPECIAL_TOKEN['MISSING']
    num_bins = int(vocab_info['num_bins'])
    loinc2tokenid = vocab_info['loinc_to_token_id']

    df = df.copy()
    df['date'] = pd.to_datetime(df['charttime']).dt.date
    df_sorted = df.sort_values(['patient_id', 'date', 'charttime'])

    patient_daily_seq = {}
    missing_value_count = 0
    total_records = 0

    grouped = df_sorted.groupby(['patient_id', 'date'])
    for (patient_id, date), day_data in grouped:
        loinc_codes = day_data['loinc_code'].astype(str).values
        vals = day_data['value'].values

        loinc_tokens = [int(CLS_TOKEN)] + [int(loinc2tokenid[str(lc)]) for lc in loinc_codes]
        value_tokens = [int(PAD_TOKEN)]
        missing_mask = [1.0]

        for v in vals:
            total_records += 1
            if pd.isna(v):
                value_tokens.append(int(MISSING_TOKEN))
                missing_mask.append(0.0)
                missing_value_count += 1
            else:
                b = int(v)
                if b < 0 or b >= num_bins:
                    b = max(0, min(num_bins - 1, b))
                value_tokens.append(4 + b)
                missing_mask.append(1.0)
        
        seq_key = (str(patient_id), str(date))
        seq_len = len(loinc_tokens)
        assert seq_len == len(value_tokens) == len(missing_mask), "sequence fields must have equal length"

        patient_daily_seq[seq_key] = {
            'loinc_tokens': loinc_tokens,
            'value_tokens': value_tokens,
            'missing_mask': missing_mask,
            'sequence_length': seq_len,
            'lab_event_count': seq_len - 1,
        }

    sequence_lengths = [s['sequence_length'] for s in patient_daily_seq.values()]
    lab_counts = [s['lab_event_count'] for s in patient_daily_seq.values()]
    print("\n=== Sequence Building Complete (pre-binned) ===")
    print(f"Total sequences created: {len(patient_daily_seq)}")
    print(f"Missing lab values: {missing_value_count}/{max(total_records,1)} "
          f"({(missing_value_count/max(total_records,1))*100:.1f}%)")
    if sequence_lengths:
        print(f"Seq len min/max: {min(sequence_lengths)}/{max(sequence_lengths)}")
    if lab_counts:
        print(f"Labs per day min/max: {min(lab_counts)}/{max(lab_counts)}")

    return patient_daily_seq, {
        'missing_value_count': int(missing_value_count),
        'total_records': int(total_records)
    }
    


def main():
    """
    Build vocabulary and sequences, then save JSON + Pickle.
    Expects df['value'] to be pre-binned to 0..num_bins-1 (NaN for missing).
    """
    try:
        print("Loading data...")
        csv_path = 'data/clean/norm/normalized_data.csv'
        df = pd.read_csv(csv_path)

        # Required columns check
        required_cols = ['patient_id', 'charttime', 'loinc_code', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return

        # Create vocabulary 
        print("Creating vocabulary...")
        NUM_BINS = 10
        vocab_info = create_vocabulary(df, num_bins=NUM_BINS)
        MISSING_TOKEN = vocab_info['special_tokens']['MISSING']

        # Build sequences
        print("Building sequences...")
        patient_sequences, build_stats = build_sequence(df, vocab_info)  

        # Preview first 5 
        print("\n" + "="*60)
        print("TOKEN SEQUENCES (Preview)")
        print("="*60)
        for i, (seq_key, seq_data) in enumerate(list(patient_sequences.items())[:5]):
            patient_id, date = seq_key
            loinc_tokens = seq_data['loinc_tokens']
            value_tokens = seq_data['value_tokens']
            missing_mask = seq_data['missing_mask']
            seq_length = seq_data['sequence_length']

            print(f"\n{i+1}. Patient: {patient_id}, Date: {date}")
            print(f"   loinc_tokens: {loinc_tokens}")
            print(f"   value_tokens: {value_tokens}")
            print(f"   missing_mask: {missing_mask}")
            print(f"   length: {seq_length}, labs: {seq_data['lab_event_count']}")
            print(f"   missing count: {int(sum(1 for m in missing_mask if m == 0))}/{seq_length}")

        # Save vocabulary
        vocab_json_path = 'data/vocabulary_mimic.json'
        print("\nSaving vocabulary...")
        with open(vocab_json_path, 'w') as f:
            json.dump(vocab_info, f, indent=2, default=str)
        print(f"✓ Vocabulary saved to '{vocab_json_path}'")

        # Save sequences as pickle
        print("Saving sequences to Pickle...")
        output_pkl_path = 'data/labevents_mimic.pkl'
        output_data = {
            'patient_sequences': patient_sequences,
            'vocab_info': vocab_info,
            'MISSING_TOKEN': MISSING_TOKEN,
            'metadata': {
                'total_sequences': len(patient_sequences),
                'vocab_size': vocab_info['vocab_size'],
                'lab_event_count': vocab_info['lab_event_count'],
                'missing_value_count': build_stats['missing_value_count'],
                'total_records': build_stats['total_records'],
            }
        }
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=4)
        print(f"✓ Sequences saved to '{output_pkl_path}'")

        # Summary
        print(f"\n=== Pickle File Info ===")
        print(f"Total sequences: {len(patient_sequences):,}")
        print(f"Vocab size: {vocab_info['vocab_size']}")
        print(f"num_bins: {vocab_info['num_bins']}")
        print(f"File saved at: {output_pkl_path}")

        return vocab_info, patient_sequences

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
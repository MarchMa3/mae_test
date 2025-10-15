import pickle
import numpy as np

with open('data/labevents_mimic.pkl', 'rb') as f:
    data = pickle.load(f)

lengths = [seq['sequence_length'] for seq in data['patient_sequences'].values()]
lengths = np.array(lengths)

print(f"50th percentile (median): {int(np.percentile(lengths, 50))}")
print(f"75th percentile: {int(np.percentile(lengths, 75))}")
print(f"90th percentile: {int(np.percentile(lengths, 90))}")
print(f"95th percentile: {int(np.percentile(lengths, 95))}")
print(f"99th percentile: {int(np.percentile(lengths, 99))}")
print(f"Max: {np.max(lengths)}")
MAE for laboratory test data.

# File Structure
## Data preprocessing
`data_preprocessing.py`: 
* Loads raw data with patient IDs, timestamps, lab values, and LOINC codes
* Cleans duplicates and missing LOINC codes
* Performs quantile normalization (n_bins=10)
* Outputs cleaned data to `./data/clean/norm/`

`tokenization.py`:
* Creates vocabulary mapping LOINC codes to token IDs
* Builds daily sequences for each patient in format: `[CLS, lab1, value1, lab2, value2, ...]`
* Generates missing masks (1=observed, 0=missing)
* Saves sequences to HDF5 format (`labevents.h5`) and vocabulary to JSON

`tokenization_v2.py`: Alternative tokenization approach (Version 2)
* Separates LOINC codes and values into distinct token sequences
* Format: loinc_tokens: `[CLS, loinc1, loinc2, ...]` and value_tokens: `[0.0, value1, value2, ...]`
* Saves to `labevents_v2.h5` and `vocabulary_v2.json`

## Model Architecture
`MAE.py`:
* MAE with with attention masking
* Reconstructs masked tokens using MSE loss
* Includes model variants: `mae_small, mae_base, mae_large`

`MAE_v2.py`: Alternative MAE implementation (Version 2)
* Processes LOINC codes and values tokens separately before combining
* + `tokenization_v2.py`

## Utilities
`utils.py`: Helper functions
* Sinusoidal positional embeddings
* Learning rate scheduling with cosine decay
* Load dataset
* Save/Load chackpoints
* Random seed setting and parameter counting

`utils_v2.py`: 
* Modified `LabDataset` to handle LOINC and value tokens separately


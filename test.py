import pandas as pd
from datetime import datetime, timedelta

def process_sequences(input_file, output_file):
    """
    Read original txt file and merge sequences within 7-day time intervals
    """
    # Read data
    data = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        # Skip header and separator line
        for line in lines[2:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    data.append({
                        'Patient_ID': parts[0],
                        'Date': parts[1],
                        'Sequence_Length': int(parts[2])
                    })
    
    df = pd.DataFrame(data)
    
    # Extract patient number and date from Patient_ID
    df['Patient_Num'] = df['Patient_ID'].apply(lambda x: x.split('_')[0])
    df['Year'] = df['Patient_ID'].apply(lambda x: x.split('_')[1])
    df['Date_Str'] = df['Patient_ID'].apply(lambda x: '_'.join(x.split('_')[2:]))
    
    # Convert date string to datetime object
    df['DateTime'] = df.apply(lambda row: 
        datetime.strptime(f"{row['Year']}_{row['Date_Str']}", '%Y_%m_%d'), axis=1)
    
    # Sort by patient and time
    df = df.sort_values(['Patient_Num', 'DateTime']).reset_index(drop=True)
    
    # Create sequence groups
    results = []
    
    for patient in df['Patient_Num'].unique():
        patient_data = df[df['Patient_Num'] == patient].copy()
        
        if len(patient_data) == 0:
            continue
        
        # Initialize first sequence group
        current_group_start = patient_data.iloc[0]['DateTime']
        current_group_ids = [patient_data.iloc[0]['Patient_ID']]
        current_group_lengths = [patient_data.iloc[0]['Sequence_Length']]
        current_group_dates = [patient_data.iloc[0]['Date_Str']]
        
        for i in range(1, len(patient_data)):
            current_date = patient_data.iloc[i]['DateTime']
            time_diff = (current_date - current_group_start).days
            
            # If time difference is within 7 days, add to current group
            if time_diff <= 7:
                current_group_ids.append(patient_data.iloc[i]['Patient_ID'])
                current_group_lengths.append(patient_data.iloc[i]['Sequence_Length'])
                current_group_dates.append(patient_data.iloc[i]['Date_Str'])
            else:
                # Save current group and start new group
                results.append({
                    'Patient_ID': current_group_ids[0],
                    'Date': 'N/A',
                    'Sequence_Length': sum(current_group_lengths),
                    'Time_Range': ' + '.join(current_group_dates),
                    'Group_Size': len(current_group_ids)
                })
                
                # Start new group
                current_group_start = current_date
                current_group_ids = [patient_data.iloc[i]['Patient_ID']]
                current_group_lengths = [patient_data.iloc[i]['Sequence_Length']]
                current_group_dates = [patient_data.iloc[i]['Date_Str']]
        
        # Save last group
        results.append({
            'Patient_ID': current_group_ids[0],
            'Date': 'N/A',
            'Sequence_Length': sum(current_group_lengths),
            'Time_Range': ' + '.join(current_group_dates),
            'Group_Size': len(current_group_ids)
        })
    
    # Create result DataFrame
    result_df = pd.DataFrame(results)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("Patient_ID\tDate\tSequence_Length\tGroup_Size\tTime_Range\n")
        f.write("-" * 100 + "\n")
        for _, row in result_df.iterrows():
            f.write(f"{row['Patient_ID']}\t{row['Date']}\t{row['Sequence_Length']}\t"
                   f"{row['Group_Size']}\t{row['Time_Range']}\n")
    
    print(f"Processing complete! Results saved to {output_file}")
    print(f"\nStatistics:")
    print(f"Original records: {len(df)}")
    print(f"Merged records: {len(result_df)}")
    print(f"Merged {len(df) - len(result_df)} records")
    
    return result_df

# Usage example
if __name__ == "__main__":
    input_file = "sequences_info.txt"  
    output_file = "merged_sequences.txt"  
    
    result = process_sequences(input_file, output_file)
    
    # Display first few rows of results
    print("\nFirst 10 rows preview:")
    print(result.head(10).to_string())
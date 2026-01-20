import pandas as pd
import glob
import os

# Define paths
data_dir = "data"
output_file = os.path.join(data_dir, "wifi_har_data.csv")

# Get list of all CSV files excluding the output file if it exists
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
csv_files = [f for f in csv_files if "wifi_har_data.csv" not in f]

if not csv_files:
    print("No new CSV files found to combine.")
else:
    print(f"Combining {len(csv_files)} files: {[os.path.basename(f) for f in csv_files]}")
    
    # Read and concatenate
    df_list = []
    for f in csv_files:
        try:
            # Read csv, ensure columns are correct
            df = pd.read_csv(f)
            # Basic validation
            if {'Timestamp', 'RSSI', 'Activity'}.issubset(df.columns):
                df_list.append(df)
            else:
                print(f"Skipping {f}: Missing required columns.")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        # Drop duplicates just in case
        combined_df.drop_duplicates(inplace=True)
        # Sort by Activity then Timestamp (optional)
        combined_df.sort_values(by=['Activity', 'Timestamp'], inplace=True)
        
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully created {output_file} with {len(combined_df)} rows.")
    else:
        print("No valid data found.")

import os
import pandas as pd
from fcsparser import parse
from pathlib import Path
import glob

def convert_fcs_to_csv(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fcs_files = glob.glob(os.path.join(input_dir, "*.fcs"))
    
    print(f"Found {len(fcs_files)} FCS files to process")
    
    for fcs_file in fcs_files:
        try:
            base_name = os.path.splitext(os.path.basename(fcs_file))[0]
            
            if 'Live_cells_tight_viSNE_GBM' in base_name:
                print(f"Skipping {base_name} - not a full normalized file")
                continue
                
            output_file = os.path.join(output_dir, f"{base_name}.csv")
            
            print(f"Converting {base_name}...")
            meta, data = parse(fcs_file)
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            
            print(f"Successfully converted {base_name} to CSV")
            
        except Exception as e:
            print(f"Error converting {fcs_file}: {str(e)}")

if __name__ == "__main__":
    input_directory = "FlowRepository_FR-FCM-Z24K_files"
    output_directory = "converted_csv_files"
    
    convert_fcs_to_csv(input_directory, output_directory)
    print("Conversion complete!") 

    
    
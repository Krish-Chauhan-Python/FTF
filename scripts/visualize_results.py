import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    predictions_files = glob.glob("outputs/*_predictions.csv")
    if not predictions_files:
        print("[WARN] No prediction files found in outputs/")
        return
        
    for pfile in predictions_files:
        try:
            df = pd.read_csv(pfile)
            if 'error' not in df.columns:
                df['error'] = df['predicted'] - df['actual']
                
            plt.figure(figsize=(8, 6))
            sns.histplot(df['error'], kde=True, bins=50)
            plt.title(f"Error Distribution for {os.path.basename(pfile)}")
            plt.xlabel("Torque Error (Nm)")
            
            out_png = pfile.replace(".csv", "_dist.png")
            plt.savefig(out_png)
            plt.close()
            print(f"[INFO] Saved plot to {out_png}")
        except Exception as e:
            print(f"Error processing {pfile}: {e}")

if __name__ == "__main__":
    main()

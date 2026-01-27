import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of your 7 CSV file paths
csv_paths = [
    'path/to/file1.csv',
    'path/to/file2.csv',
    'path/to/file3.csv',
    'path/to/file4.csv',
    'path/to/file5.csv',
    'path/to/file6.csv',
    'path/to/file7.csv'
]

# Dictionary to store data from each file
data_dict = {}
colors = plt.cm.tab10(np.linspace(0, 1, 7))  # 7 distinct colors

fig, ax = plt.subplots(figsize=(12, 8))

for i, path in enumerate(csv_paths):
    try:
        # Load CSV (adjust column names if different)
        df = pd.read_csv(path)
        
        # Assume columns are 'actual' and 'predicted' - adjust if needed
        actual = df['actual'].values
        predicted = df['predicted'].values
        
        # Create x-axis as index (0 to len-1) for each dataset
        x = np.arange(len(actual))
        
        # Plot both actual and predicted for this file
        label_actual = f'File {i+1} Actual'
        label_pred = f'File {i+1} Predicted'
        
        ax.plot(x, actual, 'o-', color=colors[i], alpha=0.7, 
                label=label_actual, markersize=4, linewidth=2)
        ax.plot(x, predicted, 's--', color=colors[i], alpha=0.8, 
                label=label_pred, markersize=4, linewidth=2)
        
        data_dict[f'File {i+1}'] = {'actual': actual, 'predicted': predicted}
        
        print(f"Loaded {path}: {len(actual)} samples")
        
    except Exception as e:
        print(f"Error loading {path}: {e}")

# Customize plot
ax.set_xlabel('Sample Index')
ax.set_ylabel('Value')
ax.set_title('Actual vs Predicted Values Across 7 Datasets')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Optional: Print summary statistics
print("\nSummary Statistics:")
for filename, data in data_dict.items():
    print(f"{filename}:")
    print(f"  Actual - Mean: {data['actual'].mean():.3f}, Std: {data['actual'].std():.3f}")
    print(f"  Predicted - Mean: {data['predicted'].mean():.3f}, Std: {data['predicted'].std():.3f}")
    print()

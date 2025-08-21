import numpy as np
import pandas as pd
import json

# Import SimplifiedZKP from convert_to_zkp.py in the same directory
import importlib.util
import sys
import os

module_path = os.path.join(os.path.dirname(__file__), 'prod\convert_to_zkp.py')
spec = importlib.util.spec_from_file_location('convert_to_zkp', module_path)
convert_to_zkp = importlib.util.module_from_spec(spec)
sys.modules['convert_to_zkp'] = convert_to_zkp
spec.loader.exec_module(convert_to_zkp)
SimplifiedZKP = convert_to_zkp.SimplifiedZKP

# Load SVM model
weights = np.load('prod\svm_weights.npy')
bias = np.load('prod\svm_bias.npy')
zkp_system = SimplifiedZKP()

# Load a transaction sample (simulate sending one transaction)
df = pd.read_csv(r'prod\archive/creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# For demo, pick a random transaction
sample_id = np.random.randint(0, len(df))
sample_features = X.iloc[sample_id].values
true_label = y.iloc[sample_id]

# Generate ZKP proof for this transaction
proof = zkp_system.prove(sample_features, weights, bias)

# Prepare message to send to Bank2
message = {
    'sample_id': int(sample_id),
    'prediction': proof['prediction'],
    'proof': proof
}

# Simulate sending by writing to a file
with open('bank_message.json', 'w') as f:
    json.dump(message, f, indent=2)

print(f"Bank1: Sent transaction #{sample_id} (true label: {true_label}) with prediction: {proof['prediction']} and ZKP proof to Bank2.")

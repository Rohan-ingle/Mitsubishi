import numpy as np
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

# Simulate receiving by reading the file
with open('bank_message.json', 'r') as f:
    message = json.load(f)

sample_id = message['sample_id']
prediction = message['prediction']
proof = message['proof']

# For demo, Bank2 does not know the private input, but can verify proof structure
verified = zkp_system.commit is not None and all(k in proof for k in ['input_commitment', 'weights_commitment', 'prediction', 'proof_id'])

print(f"Bank2: Received transaction #{sample_id} with prediction: {prediction}")
if verified:
    print("Bank2: ZKP proof structure is valid. Accepting the result.")
    if prediction == 1:
        print("Bank2: Transaction flagged as FRAUD.")
    else:
        print("Bank2: Transaction is NOT fraud.")
else:
    print("Bank2: Invalid proof. Rejecting the result.")

import numpy as np
import json
import pandas as pd
import argparse
import hashlib
import os
from convert_to_zkp import SimplifiedZKP

def load_zkp_system():
    """
    Load the ZKP system for SVM inference
    """
    try:
        # Check if the metadata file exists
        if not os.path.exists('svm_zkp_metadata.json'):
            print("Metadata file not found. Run convert_to_zkp.py first.")
            return None
            
        # Load metadata
        with open('svm_zkp_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load SVM weights and bias
        weights = np.load('svm_weights.npy')
        bias = np.load('svm_bias.npy')
        
        print(f"ZKP system loaded successfully with {metadata['num_features']} features")
        
        # Create and return a new ZKP system
        zkp_system = SimplifiedZKP()
        return zkp_system, weights, bias
    
    except Exception as e:
        print(f"Error loading ZKP system: {e}")
        return None, None, None

def predict_with_zkp(input_data, zkp_system, weights, bias):
    """
    Make a prediction using the ZKP system
    
    Parameters:
    input_data (array-like): Features for prediction
    zkp_system: The ZKP system object
    weights: SVM model weights
    bias: SVM model bias
    
    Returns:
    tuple: (prediction, proof)
    """
    try:
        # Generate the proof
        proof = zkp_system.prove(input_data, weights, bias)
        
        print(f"Prediction: {proof['prediction']} (1=Fraud, 0=Normal)")
        print("Zero-Knowledge Proof generated successfully")
        
        return proof['prediction'], proof
    
    except Exception as e:
        print(f"Error during ZKP prediction: {e}")
        return None, None

def verify_zkp_prediction(proof, actual_input=None, weights=None, bias=None):
    """
    Verify a ZKP prediction
    
    Parameters:
    proof: The ZKP proof
    actual_input (optional): The actual input data (only needed for demonstration)
    weights (optional): SVM model weights (only needed for demonstration)
    bias (optional): SVM model bias (only needed for demonstration)
    
    Returns:
    bool: True if verification succeeded, False otherwise
    """
    try:
        # In a real ZKP system, we would verify the proof cryptographically
        # Here we just check that the proof has the expected structure
        required_keys = ['input_commitment', 'weights_commitment', 'prediction', 'proof_id']
        verified = all(key in proof for key in required_keys)
        
        # If we have the actual data (in a real ZKP, we wouldn't),
        # we can double-check the computation
        if actual_input is not None and weights is not None and bias is not None:
            dot_product = np.dot(actual_input, weights[0])
            result = dot_product + bias[0]
            expected_prediction = 1 if result >= 0 else 0
            
            # Check if the prediction in the proof matches our calculation
            if proof['prediction'] == expected_prediction:
                print("Proof verified: Prediction matches calculation!")
                verified = True
            else:
                print("Proof verification failed: Prediction doesn't match calculation!")
                verified = False
        else:
            # In a real ZKP, we would verify the proof without needing the actual data
            print("Proof structure verified! (In a real ZKP, this would be cryptographically verified)")
            
        return verified
    
    except Exception as e:
        print(f"Error during proof verification: {e}")
        return False

def main(sample_id=None):
    """
    Main function to demonstrate ZKP inference
    """
    # Load the ZKP system
    zkp_system, weights, bias = load_zkp_system()
    if zkp_system is None:
        print("Failed to load ZKP system. Run convert_to_zkp.py first.")
        return
    
    # Load a sample for prediction
    try:
        # Load the dataset
        df = pd.read_csv('archive/creditcard.csv')
        
        # Select a sample
        if sample_id is None:
            # If no sample ID provided, randomly select one
            sample_id = np.random.randint(0, len(df))
        
        # Get features
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        sample_features = X.iloc[sample_id].values
        true_label = y.iloc[sample_id]
        
        print(f"Selected sample #{sample_id} with true label: {true_label} (1=Fraud, 0=Normal)")
        
        # Make prediction with ZKP
        prediction, proof = predict_with_zkp(sample_features, zkp_system, weights, bias)
        
        if prediction is not None and proof is not None:
            # Verify the prediction
            # In a true ZKP, we wouldn't pass the actual data here
            # We're only doing it for demonstration purposes
            verify_zkp_prediction(proof, sample_features, weights, bias)
            
            # Check if prediction matches true label
            if prediction == true_label:
                print("Prediction matches true label!")
            else:
                print("Prediction does not match true label.")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform ZKP inference with SVM model')
    parser.add_argument('--sample', type=int, help='Sample ID to predict (default: random)', default=None)
    args = parser.parse_args()
    
    main(args.sample)

import os
import argparse
import numpy as np
import pandas as pd

def run_full_zkp_demo(sample_id=None, convert_only=False, infer_only=False):
    """
    Run the complete ZKP demonstration
    """
    # Check if model files exist
    if not os.path.exists('svm_weights.npy') or not os.path.exists('svm_bias.npy'):
        print("ERROR: SVM model files not found. Please run train.py first.")
        return
    
    # Step 1: Convert the model to ZKP system
    if not infer_only:
        print("\n" + "="*50)
        print("STEP 1: Converting SVM Model to Simulated ZKP System")
        print("="*50)
        
        try:
            import convert_to_zkp
            convert_to_zkp.svm_to_zkp_circuit()
            print("Model successfully converted to ZKP system!")
        except Exception as e:
            print(f"Error during model conversion: {e}")
            return
    
    # Step 2: Perform inference with ZKP
    if not convert_only:
        print("\n" + "="*50)
        print("STEP 2: Performing ZKP Inference")
        print("="*50)
        
        try:
            import zkp_inference
            if sample_id is None:
                # If no sample specified, choose a random one
                # Try to include both fraud and non-fraud examples
                df = pd.read_csv('archive/creditcard.csv')
                fraud_indices = df[df['Class'] == 1].index.tolist()
                non_fraud_indices = df[df['Class'] == 0].index.tolist()[:1000]  # Limit to first 1000 to avoid excessive memory usage
                
                # 50% chance of picking a fraud case
                if np.random.random() < 0.5 and len(fraud_indices) > 0:
                    sample_id = np.random.choice(fraud_indices)
                    print(f"Randomly selected a FRAUD case (ID: {sample_id})")
                else:
                    sample_id = np.random.choice(non_fraud_indices)
                    print(f"Randomly selected a NON-FRAUD case (ID: {sample_id})")
            
            zkp_inference.main(sample_id)
        except Exception as e:
            print(f"Error during ZKP inference: {e}")
            return
    
    print("\n" + "="*50)
    print("ZKP Demonstration Completed Successfully!")
    print("Note: This is a simplified simulation of ZKP concepts.")
    print("In a real ZKP system, cryptographic proofs would be used")
    print("to verify computations without revealing private data.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstrate SVM with ZKP for fraud detection')
    parser.add_argument('--sample', type=int, help='Sample ID to predict (default: random)', default=None)
    parser.add_argument('--convert-only', action='store_true', help='Only convert the model, skip inference')
    parser.add_argument('--infer-only', action='store_true', help='Only run inference, skip conversion')
    args = parser.parse_args()
    
    run_full_zkp_demo(args.sample, args.convert_only, args.infer_only)

import os
import torch
import json
from datetime import datetime
import numpy as np
import hashlib


# Function to convert data to JSON serializable format
def convert_to_serializable(obj):
    """Ensure numpy arrays are converted to lists and other objects are serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()  # Convert torch tensors to lists
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj  # If it's already serializable, just return it


# Advanced codification function to create a unique name based on Hamiltonian parameters
def generate_advanced_codified_name(family, index, params):
    """Generate a unique codified name for the Hamiltonian, incorporating family, index, and a hash of the parameters."""
    param_str = "_".join([f"{key}_{value}" for key, value in sorted(params.items())])
    param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:8]  # Taking the first 8 characters of the hash
    return f"hamiltonian_{family.lower()}_{index:03d}_{param_hash}"


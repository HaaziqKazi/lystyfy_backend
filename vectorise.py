import numpy as np 

def features_to_vector(chunk_features):
    """
    Convert chunk features dictionary to a single vector
    
    Returns: numpy array of all features flattened
    """
    vector = []
    
    # Scalar features
    vector.append(chunk_features['rms_mean'])
    vector.append(chunk_features['rms_std'])
    vector.append(chunk_features['tempo'])
    vector.append(chunk_features['num_beats'])
    vector.append(chunk_features['spectral_centroid_mean'])
    vector.append(chunk_features['spectral_centroid_std'])
    vector.append(chunk_features['spectral_bandwidth_mean'])
    vector.append(chunk_features['spectral_bandwidth_std'])
    vector.append(chunk_features['band_energy_ratio'])
    
    # Array features (flatten them)
    vector.extend(chunk_features['mfcc_mean'])  # 13 values
    vector.extend(chunk_features['mfcc_std'])  # 13 values
    vector.extend(chunk_features['mfcc_delta_mean'])  # 13 values
    vector.extend(chunk_features['chroma_mean'])  # 12 values
    
    return np.array(vector)

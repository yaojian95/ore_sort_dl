import numpy as np

def compute_R(low, high, I0_low=195, I0_high=196, input='images', method='a', const=[5, 20]):
    '''
    Compute R-value from low and high energy images or pixels.
    
    Args:
        low: Low energy image or pixel array.
        high: High energy image or pixel array.
        I0_low: Baseline intensity for low energy.
        I0_high: Baseline intensity for high energy.
        input: 'images' or 'pixels'.
        method: Computation method 'a' or 'b'.
        const: Constants for method 'a' [c1, c2].

    Returns:
        R-value array or list. If low or high input is 255 (background), the result is 255.
    '''
    
    if input == 'images':
        # Create a mask for background values (255)
        # We need to handle potential floating point inputs if they are not exactly 255 but close, 
        # but usually 255 is an integer marker for background. 
        # Assuming inputs might be float arrays if processed, but if they come from 8-bit images they are uint8 or float.
        # Let's assume strict 255 check is what is desired for "background value".
        
        mask = (low == 255) | (high == 255)
        
        if method == 'a':
            R = np.log(I0_low / (low + 1e-6) + const[0]) / np.log(I0_high / (high + 1e-6) + const[1])
        elif method == 'b':
            R = np.log((low + 1e-6)) / (np.log(high + 1e-6 + 200.0))
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Apply mask
        R[mask] = 255
        return R

    elif input == 'pixels':
        R_values = []
        # Convert to numpy array for easier handling if it's a list, but preserving 'pixels' logic loop if desired.
        # The user's original code iterated. Vectorization is better but let's stick to the structure but optimized or just fix the logic.
        # The prompt asks "ensure low or high image in background value (255) places... calculated R value is also 255".
        # For pixels input, it's a list of values.
        
        for i in range(len(low)):
            l_val = low[i]
            h_val = high[i]
            
            if l_val == 255 or h_val == 255:
                R_i = 255
            else:
                if method == 'a':
                    R_i = np.log(I0_low / (l_val + 1e-6) + const[0]) / np.log(I0_high / (h_val + 1e-6) + const[1])
                elif method == 'b':
                    R_i = np.log((l_val + 1e-6)) / (np.log(h_val + 1e-6 + 200.0))
                else:
                    raise ValueError(f"Unknown method: {method}")
            
            R_values.append(R_i)

        return R_values
    else:
        raise ValueError(f"Unknown input type: {input}")

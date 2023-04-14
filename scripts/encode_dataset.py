"""
    Encode dataset so we can train longer/faster.    
"""

import polars as pl
from segment_anything import sam_model_registry


# Load encoder
device = get_device(device)
sam_model = sam_model_registry[model](checkpoint=weights_filepath)

# For each directory

    # Load dataset

    # Go through N items of dataset

        # Encode each item

        # Save encoded dataset
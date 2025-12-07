"""Early stopping implementation for training.

Author: Oliver Perrin
Date: December 2025
"""


class EarlyStopping:
    """Stop training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss (lower is better), 'max' for accuracy
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        
    def __call__(self, metric_value: float) -> bool:
        """Check if training should stop.
        
        Args:
            metric_value: Current metric value (e.g., validation loss)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.mode == 'min':
            improved = metric_value < (self.best_value - self.min_delta)
        else:
            improved = metric_value > (self.best_value + self.min_delta)
            
        if improved:
            self.best_value = metric_value
            self.counter = 0
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            return True
            
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False

"""Gradient monitoring utilities.

Author: Oliver Perrin
Date: December 2025
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class GradientMonitor:
    """Monitor gradient statistics during training.
    
    Tracks gradient norms, helps detect gradient issues like vanishing/exploding.
    """
    
    def __init__(self, model: nn.Module, log_frequency: int = 100):
        """Initialize gradient monitor.
        
        Args:
            model: Model to monitor
            log_frequency: Log gradients every N steps
        """
        self.model = model
        self.log_frequency = log_frequency
        self.step_count = 0
        
    def compute_grad_norm(self) -> Dict[str, float]:
        """Compute gradient norm statistics.
        
        Returns:
            Dictionary with gradient statistics
        """
        total_norm = 0.0
        max_norm = 0.0
        num_params = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
                num_params += 1
        
        total_norm = total_norm ** 0.5
        
        return {
            "grad_norm": total_norm,
            "grad_norm_max": max_norm,
            "num_params_with_grad": num_params,
        }
    
    def check_gradients(self) -> Dict[str, int]:
        """Check for gradient issues (NaN, Inf, zero).
        
        Returns:
            Dictionary with counts of gradient issues
        """
        nan_count = 0
        inf_count = 0
        zero_count = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    nan_count += 1
                if torch.isinf(p.grad).any():
                    inf_count += 1
                if (p.grad == 0).all():
                    zero_count += 1
        
        return {
            "nan_grads": nan_count,
            "inf_grads": inf_count,
            "zero_grads": zero_count,
        }
    
    def log_gradients(self, step: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Log gradient statistics if it's time.
        
        Args:
            step: Current training step (uses internal counter if None)
            
        Returns:
            Gradient statistics if logged, None otherwise
        """
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        if step % self.log_frequency == 0:
            stats = self.compute_grad_norm()
            issues = self.check_gradients()
            
            # Combine stats
            all_stats = {**stats, **issues}
            
            return all_stats
        
        return None

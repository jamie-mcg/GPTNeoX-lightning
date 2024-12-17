import os
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from torchmetrics import Metric
import torch


@dataclass
class TrainingOptions:
    optimizer: str
    optimizer_args: Optional[Dict]
    lr_scheduler: str
    lr_scheduler_args: Optional[Dict]
    pl_lr_scheduler_args: Optional[Dict]
    metrics: Optional[List[Metric]]
    
@dataclass
class InferenceOptions:
    metrics: Optional[List[Metric]]
    num_samples: int = 1024

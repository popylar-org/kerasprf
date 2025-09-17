
from dataclasses import dataclass

import numpy as np

@dataclass
class Stimulus:
    dimensions: list[str]
    paradigm: np.ndarray
    coordinates: np.ndarray

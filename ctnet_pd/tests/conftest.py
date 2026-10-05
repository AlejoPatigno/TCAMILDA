import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import pytest

from ctnet_pd.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def toy_segments():
    """30 subjects (15 PD), 2 recordings x 3 segments each, two task families."""
    rows = []
    rid = 0
    for s in range(30):
        label = int(s < 15)
        for r in range(2):
            for j in range(3):
                rows.append({"subject_id": f"S{s:02d}", "label": label, "recording_id": rid,
                             "task_family": "ddk" if r == 0 else "monologue", "segment_idx": j})
            rid += 1
    return pd.DataFrame(rows)

from typing import List, Optional

import numpy as np
import pandas as pd
import os

import constants as constants


# constants to find the dataset
DIFF_DB_REPO = 'poloclub/diffusiondb'
DIFF_DB_SMALL_FILE = 'metadata.parquet'
DIFF_DB_LARGE_FILE = 'metadata-large.parquet'

# prompt to use for debugging
DEBUG_PROMPT = "giant cannon in a city"


class DiffusionDBLoader:

    def __init__(
        self,
        split: str,
        debug: Optional[bool]=False
    ):
        """ A loader for the DiffusionDB dataset.
        We use this instead of the datasets library because
        it is faster to load parquets ourselves.

        Args:
            split (str): use 'small' or 'large' to specify the dataset size
            debug (bool, optional): Load the same prompt every time. Defaults to False.
        """

        # handle inputs
        self.debug = debug
        if split.lower() == 'small':
            self.file = DIFF_DB_SMALL_FILE
        elif split.lower() == 'large':
            self.file = DIFF_DB_LARGE_FILE
        else:
            raise ValueError(f"split must be 'small' or 'large', got {split}")

        # handle debug mode
        if self.debug:
            self.data = np.array(["giant cannon in a city"])

        # handle split
        else:

            # try loading parquet locally, otherwise load from HuggingFace and save
            path = os.path.join(constants.LOCAL_DATA_DIR, DIFF_DB_REPO, self.file)
            if os.path.exists(path):
                df = pd.read_parquet(path)
            else:
                url = f"hf://datasets/{DIFF_DB_REPO}/{self.file}"
                df = pd.read_parquet(url)

                os.makedirs(os.path.dirname(path), exist_ok=True)
                df.to_parquet(path)

            self.data = np.array(df["prompt"])

        # set up the iterator
        self.curr = 0


    def __len__(self) -> int:
        # number of prompts in the dataset
        return len(self.data)

    def reset(self):
        # reset the iterator
        self.curr = 0


    def __call__(self, bs: int) -> List[str]:
        """ Get a batch of prompts from the dataset.

        Args:
            bs (int): batch size 

        Returns:
            List[str]: batch of prompts of length bs
        """
        x = []

        for _ in range(bs):
            x.append(self.data[self.curr])

            # increment the iterator
            self.curr = (self.curr + 1) % len(self.data)

        return x
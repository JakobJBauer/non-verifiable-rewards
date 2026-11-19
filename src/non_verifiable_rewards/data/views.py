"""Prompt-level views and partitions for policy training and evaluation."""

import math
import random
from collections import defaultdict

from datasets import Dataset, DatasetDict


def _prompt_groups(pairs):
    groups = defaultdict(list)
    # Group by text as well as task: source IDs can alias the same prompt.
    for index, row in enumerate(pairs.select_columns(["dataset", "prompt"])):
        groups[row["dataset"], row["prompt"].strip()].append(index)
    return groups


def split_by_prompt(
    pairs: Dataset,
    *,
    test_size: float = 0.2,
    seed: int = 1729,
) -> DatasetDict:
    """Partition pairs into train/test without splitting a prompt's stories.

    ``test_size`` is a fraction of unique prompts, not pairs. Assignment is
    independent of row order. All candidates and constructed references for a
    prompt remain in one partition, so splitting after reference construction
    is safe. Call again on train to reserve validation prompts if needed.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be a fraction strictly between 0 and 1")
    groups = _prompt_groups(pairs)
    keys = sorted(groups)
    count = math.ceil(len(keys) * test_size)
    if not 0 < count < len(keys):
        raise ValueError(
            "Need enough unique prompts for nonempty train and test splits"
        )
    random.Random(seed).shuffle(keys)
    held_out = set(keys[:count])
    return DatasetDict(
        {
            split: pairs.select(
                sorted(
                    index
                    for key, indices in groups.items()
                    if (key in held_out) == is_test
                    for index in indices
                )
            )
            for split, is_test in (("train", False), ("test", True))
        }
    )


def to_prompt_dataset(pairs: Dataset | DatasetDict) -> Dataset | DatasetDict:
    """Keep one prompt/reference for rollouts, without preference candidates.

    Repeated comparisons must not implicitly give a prompt extra RL weight.
    Choose the pair with the smallest stable pair ID for each prompt and retain
    its reference. For LitBench this fixes one of the pair-specific third-story
    proxies across rollouts/rewards; it does not establish a gold answer.
    ``source_pair_id`` records the choice. Supply only ``prompt`` to the policy;
    ``reference`` is reward-side data. Partition prompts before using this view.
    """
    if isinstance(pairs, DatasetDict):
        return DatasetDict(
            {name: to_prompt_dataset(data) for name, data in pairs.items()}
        )
    pair_ids = pairs["pair_id"]
    selected = [
        min(indices, key=lambda index: pair_ids[index])
        for _, indices in sorted(_prompt_groups(pairs).items())
    ]
    return (
        pairs.select(selected)
        .select_columns(
            [
                "dataset",
                "prompt_id",
                "prompt",
                "reference",
                "reference_id",
                "pair_id",
            ]
        )
        .rename_column("pair_id", "source_pair_id")
    )

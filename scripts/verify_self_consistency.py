"""Four-step GPU acceptance check on synthetic fixtures, one per supported task.

Run on eidf-dev with PYTHONPATH=src. This is a software smoke test, not a
dataset-quality benchmark. The supplied output directory must be fresh.
"""

import argparse
import json
from pathlib import Path

from non_verifiable_rewards.experiments.io import read_jsonl, write_json, write_jsonl
from non_verifiable_rewards.rewards.prompts import multiple_choice_prompt
from non_verifiable_rewards.training.assets import policy_path
from non_verifiable_rewards.training.config import validate_config
from non_verifiable_rewards.training.data import prepare
from non_verifiable_rewards.training.train import load_tokenizer, train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="/root/.cache/non-verifiable-rewards")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Audit an existing completed smoke run without training again",
    )
    parser.add_argument(
        "--backend", choices=("transformers", "vllm"), default="transformers"
    )
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.verify_only:
        config = json.loads((root / "config.json").read_text())
        target = Path(config["output_dir"]) / "runs" / "self_consistency"
        verify(root, config, json.loads((target / "trained.json").read_text()))
        return
    root.mkdir(parents=True, exist_ok=False)
    fixtures = {
        "reddit_tldr": (
            (
                "I missed the last bus after working late. My neighbour drove me home, "
                "and I baked her a cake to say thank you. Please summarize in one sentence."
            ),
            "After missing the bus, I thanked my neighbour with a cake for driving me home.",
        ),
        "litbench": (
            "Write a story of at most 60 words about a lighthouse keeper who hears a familiar voice on the radio.",
            "The keeper heard his own voice, warning him of tomorrow's storm.",
        ),
        "wmt14_de_en": (
            "Translate the following German text into English:\n\nDer Zug hat zehn Minuten Verspätung. Bitte warten Sie auf Gleis zwei.",
            "The train is ten minutes late. Please wait on platform two.",
        ),
        "supergpqa": (
            multiple_choice_prompt(
                "What is seven plus eight?", ["14", "15", "16", "17"]
            ),
            "15",
        ),
    }
    local = {}
    for dataset, (prompt, reference) in fixtures.items():
        local[dataset] = {}
        for split in ("train", "test"):
            row = {
                "dataset": dataset,
                "prompt_id": f"{dataset}-{split}",
                "example_id": f"{dataset}-{split}",
                "pair_id": f"{dataset}-{split}",
                "prompt": prompt if split == "train" else "Held-out fixture: " + prompt,
                "reference": reference,
                "reference_id": f"ref-{dataset}-{split}",
                "metadata": {"split": split, "synthetic_smoke_fixture": True},
            }
            if dataset == "supergpqa":
                row.update(options=["14", "15", "16", "17"], answer_letter="B")
            elif split == "test":
                row.update(preferred=reference, rejected="An irrelevant response.")
            path = root / "fixtures" / f"{dataset}-{split}.jsonl"
            write_jsonl(path, [row])
            local[dataset][split] = str(path)
    config = validate_config(
        {
            "output_dir": str(root / "sweep"),
            "cache_dir": args.cache,
            "policy": {
                "id": args.model,
                "device": "cuda",
                "dtype": "bfloat16",
                "lora": True,
                "enable_thinking": False,
            },
            "inference": {"backend": args.backend},
            "data": {
                "datasets": list(fixtures),
                "train_examples_per_dataset": 1,
                "max_prompt_tokens": 512,
                "local": local,
            },
            "optimization": {
                "max_steps": 4,
                "num_generations": 8,
                "max_completion_length": 128,
                "save_steps": 2,
                "logging_steps": 1,
            },
            "benchmark": {
                "enabled": False,
                "pairs_per_dataset": 1,
                "generation_prompts_per_dataset": 1,
                "smart_bleurt_prompts": 1,
                "rewards": ["length"],
                "max_new_tokens": 128,
            },
            "runs": [{"name": "self_consistency", "rewards": {"self_consistency": 1}}],
        }
    )
    write_json(root / "config.json", config)
    prepare(config, load_tokenizer(policy_path(config)))
    metadata = train_run(config, config["runs"][0])
    verify(root, config, metadata)


def verify(root, config, metadata):
    target = Path(config["output_dir"]) / "runs" / "self_consistency"
    traces = read_jsonl(target / "reward_trace.jsonl")
    assert len(traces) == 4
    assert {trace["datasets"][0] for trace in traces} == set(config["data"]["datasets"])
    assert [trace["step"] for trace in traces] == list(range(4))
    summary = []
    for trace in traces:
        assert len(trace["total"]) == 8 and set(trace["total"]) <= {0.0, 1.0}
        (group,) = trace["self_consistency"]
        assert group["policy_step"] == trace["step"]
        assert len(group["votes"]) == 8
        summary.append({**group, "rewards": trace["total"]})
    assert any("selected_labels" in group for group in summary)
    assert any(not group["zero_variance"] for group in summary)
    state = json.loads((target / "checkpoints" / "trainer_state.json").read_text())
    assert any(item.get("grad_norm", 0) > 0 for item in state["log_history"])
    assert metadata["steps"] == 4 and metadata["online_rewards"] == ["self_consistency"]
    assert not metadata["reward_models_frozen"]
    # The completed-run path verifies exported checksums and must not perform
    # another update or selection call.
    assert train_run(config, config["runs"][0]) == metadata
    assert len(read_jsonl(target / "reward_trace.jsonl")) == 4
    write_json(
        root / "result.json",
        {
            "passed": True,
            "scope": "Synthetic four-task GPU software smoke test",
            "backend": config["inference"]["backend"],
            "steps": 4,
            "rollouts": 32,
            "groups": summary,
            "metrics": state["log_history"],
            "export_and_completed_resume_verified": True,
        },
    )
    print(json.dumps({"passed": True, "result": str(root / "result.json")}))


if __name__ == "__main__":
    main()

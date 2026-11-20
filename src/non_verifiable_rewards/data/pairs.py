"""Build auditable preference pairs with a reference outside each pair."""

from collections import Counter, defaultdict

from .._hashing import digest

_NEWS_SITES = {"cnn", "dailymail"}


def make_pair(
    dataset, prompt_id, prompt, preferred, rejected, reference, reference_id, metadata
):
    prompt, preferred, rejected, reference = map(
        _text, (prompt, preferred, rejected, reference)
    )
    if not all((prompt, preferred, rejected, reference)):
        raise ValueError("Empty benchmark field")
    if preferred == rejected or reference in (preferred, rejected):
        raise ValueError("Candidates and the held-out reference must be distinct")
    return {
        "dataset": dataset,
        "prompt_id": prompt_id,
        "pair_id": digest([dataset, prompt_id, preferred, rejected])[:24],
        "prompt": prompt,
        "preferred": preferred,
        "rejected": rejected,
        "reference": reference,
        "reference_id": reference_id,
        "preferred_id": digest([prompt_id, preferred])[:24],
        "rejected_id": digest([prompt_id, rejected])[:24],
        "metadata": metadata,
    }


def reddit_pairs(rows, references, *, source_split=None, excluded_posts=()):
    audit = Counter()
    pairs = {}
    for index, row in enumerate(rows):
        audit["source_rows"] += 1
        info = row["info"]
        if source_split is not None and row.get("split") != source_split:
            audit["excluded_other_split"] += 1
            continue
        if _text(info.get("post")) in excluded_posts:
            audit["excluded_test_post_overlap"] += 1
            continue
        if _text(info.get("site")).lower() in _NEWS_SITES or not _text(
            info.get("post")
        ):
            audit["excluded_non_reddit"] += 1
            continue
        gold = references.get(info["id"])
        if gold is None:
            audit["excluded_missing_reference"] += 1
            continue
        if _text(gold["post"]) != _text(info["post"]):
            audit["excluded_source_post_mismatch"] += 1
            continue
        summaries, choice = row["summaries"], row["choice"]
        if len(summaries) != 2 or choice not in (0, 1):
            audit["excluded_invalid_choice"] += 1
            continue
        try:
            pair = make_pair(
                "reddit_tldr",
                info["id"],
                _sff_prompt(info),
                summaries[choice]["text"],
                summaries[1 - choice]["text"],
                gold["summary"],
                info["id"] + ":human_tldr",
                {
                    "source_row": row.get("_source_row", index),
                    "split": row.get("split"),
                    "batch": row.get("batch"),
                    "confidence": (row.get("extra") or {}).get("confidence"),
                    "preferred_policy": summaries[choice].get("policy"),
                    "rejected_policy": summaries[1 - choice].get("policy"),
                    "reference_rule": "Original human TL;DR joined by post ID; source post must match",
                },
            )
        except ValueError:
            audit["excluded_empty_or_overlapping_texts"] += 1
            continue
        if pair["pair_id"] in pairs:
            audit["excluded_duplicate_labeled_pair"] += 1
            continue
        pairs[pair["pair_id"]] = pair
    audit["eligible_pairs"] = len(pairs)
    audit["eligible_prompts"] = len({p["prompt_id"] for p in pairs.values()})
    return list(pairs.values()), dict(audit)


def litbench_pairs(rows):
    audit = Counter()
    stories = defaultdict(dict)
    for row in rows:
        prompt = _text(row["prompt"])
        for side in ("chosen", "rejected"):
            text = _text(row[side + "_story"])
            if not text:
                continue
            votes = int(row[side + "_upvotes"])
            values = stories[prompt].setdefault(text, set())
            values.add(votes)
    audit["source_prompts"] = len(stories)
    audit["prompts_with_at_least_three_stories"] = sum(
        len(group) >= 3 for group in stories.values()
    )
    audit["stories_with_multiple_observed_vote_counts"] = sum(
        len(votes) > 1 for group in stories.values() for votes in group.values()
    )
    pairs = {}
    for index, row in enumerate(rows):
        audit["source_rows"] += 1
        prompt = _text(row["prompt"])
        group = stories[prompt]
        if len(group) < 3:
            audit["excluded_fewer_than_three_stories"] += 1
            continue
        audit["rows_with_at_least_three_stories"] += 1
        if row["chosen_upvotes"] <= row["rejected_upvotes"]:
            audit["excluded_non_strict_upvote_preference"] += 1
            continue
        preferred, rejected = _text(row["chosen_story"]), _text(row["rejected_story"])
        candidates = [
            text for text in group if text and text not in (preferred, rejected)
        ]
        if not candidates:
            audit["excluded_missing_held_out_story"] += 1
            continue
        # Counts can vary across scrapes. Freeze the maximum observed count for
        # each exact story; break vote ties by a stable content hash.
        reference = min(candidates, key=lambda text: (-max(group[text]), digest(text)))
        prompt_id = digest(["litbench", prompt])[:24]
        try:
            pair = make_pair(
                "litbench",
                prompt_id,
                prompt,
                preferred,
                rejected,
                reference,
                digest([prompt_id, reference])[:24],
                {
                    "source_row": row.get("_source_row", index),
                    "preferred_upvotes": int(row["chosen_upvotes"]),
                    "rejected_upvotes": int(row["rejected_upvotes"]),
                    "reference_upvotes": max(group[reference]),
                    "stories_for_prompt": len(group),
                    "reference_observed_upvotes": sorted(group[reference]),
                    "reference_rule": "Highest maximum-observed upvotes outside the pair; hash breaks ties",
                },
            )
        except ValueError:
            audit["excluded_empty_or_overlapping_texts"] += 1
            continue
        if pair["pair_id"] in pairs:
            audit["excluded_duplicate_labeled_pair"] += 1
            continue
        pairs[pair["pair_id"]] = pair
    audit["eligible_pairs"] = len(pairs)
    audit["eligible_prompts"] = len({p["prompt_id"] for p in pairs.values()})
    return list(pairs.values()), dict(audit)


def _sff_prompt(info: dict) -> str:
    site = _text(info.get("site")).lower()
    article = _text(info.get("article"))
    post = _text(info.get("post"))
    if site in _NEWS_SITES or (article and not post):
        return f"Article:\n{article}\n\nTL;DR:"

    subreddit = _text(info.get("subreddit"))
    title = _text(info.get("title"))
    return f"SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"


def _text(value: str | None) -> str:
    return (value or "").strip()

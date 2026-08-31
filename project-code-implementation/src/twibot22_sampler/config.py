"""Static configuration for the sampling pipeline."""

from __future__ import annotations

from dataclasses import dataclass

PRESET_SIZES = {
    "smoke": 2000,
    "main": 10000,
}

POOL_SPLIT = {
    "primary": 0.85,
    "sparse": 0.15,
}

PRIMARY_POOL_MIN_TWEETS = 8
SPARSE_POOL_MIN_TWEETS = 1

DEFAULT_SEED = 42

DEFAULT_MAX_CONTEXT_MUTUAL = 10
DEFAULT_MAX_CONTEXT_FOLLOWER = 5
DEFAULT_MAX_CONTEXT_FOLLOWING = 5

DEFAULT_SEED_USER_MAX_TWEETS = 20
DEFAULT_CONTEXT_USER_MAX_TWEETS = 8
POST_PREFILTER_MULTIPLIER = 5

USER_PROFILE_COLUMNS = [
    "user_id",
    "split",
    "label",
    "verified_bucket",
    "followers_count",
    "following_count",
    "tweet_count_hint",
    "degree_proxy",
    "primary_pool",
    "sparse_pool",
]

USER_JSON_BASENAME = "user.json"
EDGE_CSV_BASENAME = "edge.csv"
LABEL_CSV_BASENAME = "label.csv"
SPLIT_CSV_BASENAME = "split.csv"

EMPTY_EXPORT_FILENAMES = ("list.jsonl", "hashtag.jsonl")

WRAPPER_KEYS = ("data", "items", "records", "users", "tweets")

USER_USER_RELATION_ALIASES = {
    "following": "following",
    "follower": "follower",
    "followers": "follower",
    "followed": "follower",
}

POST_RELATION_ALIASES = {
    "post": "post",
}

TWEET_TWEET_RELATION_ALIASES = {
    "retweet": "retweet",
    "retweeted": "retweet",
    "quote": "quote",
    "quoted": "quote",
    "reply": "reply",
    "replied": "reply",
}

MENTION_RELATION_ALIASES = {
    "mention": "mention",
    "mentioned": "mention",
}

SUPPORTED_RELATIONS = (
    tuple(USER_USER_RELATION_ALIASES.keys())
    + tuple(POST_RELATION_ALIASES.keys())
    + tuple(TWEET_TWEET_RELATION_ALIASES.keys())
    + tuple(MENTION_RELATION_ALIASES.keys())
)

OUTPUT_RELATIONS = ("following", "post", "retweet", "quote", "reply", "mention")

VALIDATE_MAX_SPLIT_RATIO_DELTA = 0.015
VALIDATE_MAX_LABEL_RATIO_DELTA = 0.015

TWEET_BUDGETS = {
    "smoke": 40000,
    "main": 250000,
}


@dataclass(frozen=True)
class SamplingThresholds:
    """Sampling hyperparameters exposed via CLI."""

    max_context_mutual: int = DEFAULT_MAX_CONTEXT_MUTUAL
    max_context_follower: int = DEFAULT_MAX_CONTEXT_FOLLOWER
    max_context_following: int = DEFAULT_MAX_CONTEXT_FOLLOWING
    seed_user_max_tweets: int = DEFAULT_SEED_USER_MAX_TWEETS
    context_user_max_tweets: int = DEFAULT_CONTEXT_USER_MAX_TWEETS


def preset_size(name: str) -> int:
    """Resolve a preset to its configured size."""

    try:
        return PRESET_SIZES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported preset: {name}") from exc

"""Command-line interface for the TwiBot-22 sampling pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .audit import run_field_audit
from .community_detection import (
    DEFAULT_COMMUNITY_ALGORITHM,
    DEFAULT_COMMUNITY_MAX_ITERATIONS,
    DEFAULT_COMMUNITY_MIN_SIZE,
    DEFAULT_COMMUNITY_MUTUAL_SUPPORT_BONUS,
    detect_communities,
)
from .community_error_analysis import (
    DEFAULT_ERROR_ANALYSIS_SPLIT,
    DEFAULT_ERROR_ANALYSIS_TOP_K,
    analyze_community_errors,
)
from .community_reranker import (
    DEFAULT_RERANKER_EARLY_STOPPING,
    DEFAULT_RERANKER_L2,
    DEFAULT_RERANKER_LEARNING_RATE,
    DEFAULT_RERANKER_MAX_EPOCHS,
    DEFAULT_RERANKER_THRESHOLD_VALUES,
    train_community_reranker,
)
from .community_reranker_analysis import (
    DEFAULT_RERANKER_ANALYSIS_SPLIT,
    DEFAULT_RERANKER_ANALYSIS_TOP_K,
    analyze_community_reranker,
)
from .community_evaluation import (
    DEFAULT_EVAL_SMOOTHING_ALPHA,
    DEFAULT_EVAL_THRESHOLD,
    evaluate_communities,
)
from .community_purity import (
    DEFAULT_PURITY_SMOOTHING_ALPHA,
    DEFAULT_PURITY_THRESHOLD,
    evaluate_community_purity,
)
from .community_finalize import (
    DEFAULT_FINALIZE_TOP_COMMUNITIES,
    finalize_best_community_run,
)
from .community_structure import analyze_community_structure
from .community_sweep import (
    DEFAULT_SWEEP_OBJECTIVE_METRIC,
    DEFAULT_SWEEP_OBJECTIVE_SPLIT,
    sweep_community_pipeline,
)
from .external_baseline_common import (
    DEFAULT_BASELINE_SEED,
    DEFAULT_LR_C_VALUES,
    DEFAULT_NODE2VEC_P_VALUES,
    DEFAULT_NODE2VEC_Q_VALUES,
    DEFAULT_RF_ESTIMATORS,
    DEFAULT_RF_MAX_DEPTHS,
    DEFAULT_WALK_DIMENSION,
    DEFAULT_WALK_EPOCHS,
    DEFAULT_WALK_LENGTH,
    DEFAULT_WALK_WINDOW,
    DEFAULT_NUM_WALKS,
)
from .external_baseline_summary import summarize_external_baselines
from .external_feature_baselines import run_feature_baselines
from .external_graph_baselines import run_graph_baselines
from .grouping_baseline_summary import summarize_grouping_baselines
from .grouping_baselines import DEFAULT_KMEANS_K_VALUES, run_kmeans_grouping_baseline
from .context_sampling import expand_context_users
from .downsample import downsample_exported_sample
from .export import collect_user_records, export_sample_dataset
from .llm_client import OpenAICompatibleClient, load_llm_settings
from .post_types import run_post_type_classification
from .profile import build_user_profile, load_profile_rows
from .readers import (
    read_label_map,
    read_split_map,
    resolve_edge_path,
    resolve_label_path,
    resolve_split_path,
    resolve_tweet_paths,
)
from .tweet_sampling import (
    collect_post_candidates_and_user_edges,
    collect_third_pass_edges,
    expand_referenced_tweets,
    finalize_post_selection,
    load_tweet_records,
)
from .triplets import run_triplet_extraction
from .temporal_profiles import DEFAULT_TEMPORAL_MIN_TWEETS, build_temporal_profiles
from .user_features import build_user_feature_table
from .user_graph import (
    DEFAULT_GRAPH_BACKEND,
    DEFAULT_GRAPH_CHUNK_SIZE,
    DEFAULT_GRAPH_CANDIDATE_K,
    DEFAULT_GRAPH_FUSION_MODE,
    DEFAULT_GRAPH_K,
    DEFAULT_GRAPH_LAMBDA_BEHAVIOR,
    DEFAULT_GRAPH_LAMBDA_CONTENT,
    DEFAULT_GRAPH_LAMBDA_NETWORK,
    DEFAULT_GRAPH_LAMBDA_TEMPORAL,
    DEFAULT_GRAPH_METRIC,
    DEFAULT_GRAPH_SYMMETRIZE,
    build_user_graph,
)
from .user_vectors import DEFAULT_EMBEDDING_MODEL, build_user_vectors
from .user_sampling import select_seed_users
from .validate import validate_sample


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TwiBot-22 sampling pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile", help="Build the user sampling profile")
    profile_parser.add_argument("--data-root", required=True, type=Path)
    profile_parser.add_argument("--work-root", required=True, type=Path)
    profile_parser.set_defaults(func=run_profile)

    sample_parser = subparsers.add_parser("sample", help="Generate a sampled subset")
    sample_parser.add_argument("--preset", required=True, choices=sorted(config.PRESET_SIZES))
    sample_parser.add_argument("--data-root", required=True, type=Path)
    sample_parser.add_argument("--work-root", required=True, type=Path)
    sample_parser.add_argument("--output-root", required=True, type=Path)
    sample_parser.add_argument("--seed", default=config.DEFAULT_SEED, type=int)
    sample_parser.add_argument("--workers", default=1, type=int)
    sample_parser.add_argument("--max-context-mutual", default=config.DEFAULT_MAX_CONTEXT_MUTUAL, type=int)
    sample_parser.add_argument("--max-context-follower", default=config.DEFAULT_MAX_CONTEXT_FOLLOWER, type=int)
    sample_parser.add_argument("--max-context-following", default=config.DEFAULT_MAX_CONTEXT_FOLLOWING, type=int)
    sample_parser.add_argument("--seed-user-max-tweets", default=config.DEFAULT_SEED_USER_MAX_TWEETS, type=int)
    sample_parser.add_argument(
        "--context-user-max-tweets",
        default=config.DEFAULT_CONTEXT_USER_MAX_TWEETS,
        type=int,
    )
    sample_parser.set_defaults(func=run_sample)

    validate_parser = subparsers.add_parser("validate", help="Validate an exported sample")
    validate_parser.add_argument("--sample-root", required=True, type=Path)
    validate_parser.add_argument("--report-out", required=False, type=Path)
    validate_parser.set_defaults(func=run_validate)

    audit_parser = subparsers.add_parser("audit", help="Audit field availability on a sampled subset")
    audit_parser.add_argument("--sample-root", required=True, type=Path)
    audit_parser.add_argument("--output-root", required=False, type=Path)
    audit_parser.add_argument("--min-triplet-tweets", default=8, type=int)
    audit_parser.add_argument("--min-time-tweets", default=8, type=int)
    audit_parser.add_argument("--min-behavior-tweets", default=1, type=int)
    audit_parser.set_defaults(func=run_audit)

    triplet_parser = subparsers.add_parser("extract-triplets", help="Run LLM triplet compression on sampled tweets")
    triplet_parser.add_argument("--sample-root", required=True, type=Path)
    triplet_parser.add_argument("--output-root", required=False, type=Path)
    triplet_parser.add_argument("--seed", default=config.DEFAULT_SEED, type=int)
    triplet_parser.add_argument("--per-user-limit", default=20, type=int)
    triplet_parser.add_argument("--min-user-tweets", default=8, type=int)
    triplet_parser.add_argument("--max-users", required=False, type=int)
    triplet_parser.add_argument("--max-tweets", required=False, type=int)
    triplet_parser.add_argument("--overwrite", action="store_true")
    _add_llm_args(triplet_parser)
    triplet_parser.set_defaults(func=run_extract_triplets)

    post_type_parser = subparsers.add_parser(
        "classify-post-types",
        help="Run post-type classification on sampled tweets",
    )
    post_type_parser.add_argument("--sample-root", required=True, type=Path)
    post_type_parser.add_argument("--output-root", required=False, type=Path)
    post_type_parser.add_argument("--mode", choices=("heuristic", "hybrid", "llm"), default="hybrid")
    post_type_parser.add_argument("--seed", default=config.DEFAULT_SEED, type=int)
    post_type_parser.add_argument("--per-user-limit", default=20, type=int)
    post_type_parser.add_argument("--min-user-tweets", default=1, type=int)
    post_type_parser.add_argument("--max-users", required=False, type=int)
    post_type_parser.add_argument("--max-tweets", required=False, type=int)
    post_type_parser.add_argument("--overwrite", action="store_true")
    _add_llm_args(post_type_parser)
    post_type_parser.set_defaults(func=run_classify_post_types)

    downsample_parser = subparsers.add_parser(
        "downsample-final",
        help="Create a smaller derived sample from an exported sample",
    )
    downsample_parser.add_argument("--sample-root", required=True, type=Path)
    downsample_parser.add_argument("--output-root", required=True, type=Path)
    downsample_parser.add_argument("--target-users", required=True, type=int)
    downsample_parser.add_argument("--seed", default=config.DEFAULT_SEED, type=int)
    downsample_parser.set_defaults(func=run_downsample_final)

    feature_parser = subparsers.add_parser(
        "build-user-features",
        help="Build a unified user-level feature table from sampled and derived outputs",
    )
    feature_parser.add_argument("--sample-root", required=True, type=Path)
    feature_parser.add_argument("--output-root", required=False, type=Path)
    feature_parser.add_argument("--triplet-seed", default=config.DEFAULT_SEED, type=int)
    feature_parser.add_argument("--post-type-seed", default=config.DEFAULT_SEED, type=int)
    feature_parser.set_defaults(func=run_build_user_features)

    vector_parser = subparsers.add_parser(
        "build-user-vectors",
        help="Build user text embeddings, numeric vectors, and fused vectors",
    )
    vector_parser.add_argument("--sample-root", required=True, type=Path)
    vector_parser.add_argument("--feature-root", required=False, type=Path)
    vector_parser.add_argument("--output-root", required=False, type=Path)
    vector_parser.add_argument("--embedding-model", required=False)
    vector_parser.add_argument("--text-field", default="triplet_document")
    vector_parser.add_argument("--batch-size", default=64, type=int)
    vector_parser.add_argument("--max-users", required=False, type=int)
    vector_parser.add_argument("--no-description-fallback", action="store_true")
    vector_parser.add_argument("--base-url", required=False)
    vector_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    vector_parser.add_argument("--env-file", required=False, type=Path)
    vector_parser.add_argument("--requests-per-minute", required=False, type=int)
    vector_parser.add_argument("--request-timeout-s", default=60.0, type=float)
    vector_parser.add_argument("--max-retries", default=3, type=int)
    vector_parser.set_defaults(func=run_build_user_vectors)

    temporal_parser = subparsers.add_parser(
        "build-temporal-profiles",
        help="Build UTC-hour temporal posting profiles for sampled users",
    )
    temporal_parser.add_argument("--sample-root", required=True, type=Path)
    temporal_parser.add_argument("--output-root", required=False, type=Path)
    temporal_parser.add_argument("--min-time-tweets", default=DEFAULT_TEMPORAL_MIN_TWEETS, type=int)
    temporal_parser.set_defaults(func=run_build_temporal_profiles)

    graph_parser = subparsers.add_parser(
        "build-user-graph",
        help="Build a user similarity graph via early or late fusion",
    )
    graph_parser.add_argument("--sample-root", required=True, type=Path)
    graph_parser.add_argument("--vector-root", required=False, type=Path)
    graph_parser.add_argument("--feature-root", required=False, type=Path)
    graph_parser.add_argument("--temporal-root", required=False, type=Path)
    graph_parser.add_argument("--output-root", required=False, type=Path)
    graph_parser.add_argument("--k", default=DEFAULT_GRAPH_K, type=int)
    graph_parser.add_argument("--metric", choices=(DEFAULT_GRAPH_METRIC,), default=DEFAULT_GRAPH_METRIC)
    graph_parser.add_argument("--min-similarity", default=0.0, type=float)
    graph_parser.add_argument(
        "--fusion-mode",
        choices=("early", "late"),
        default=DEFAULT_GRAPH_FUSION_MODE,
    )
    graph_parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "python"),
        default=DEFAULT_GRAPH_BACKEND,
    )
    graph_parser.add_argument("--chunk-size", default=DEFAULT_GRAPH_CHUNK_SIZE, type=int)
    graph_parser.add_argument("--candidate-k", default=DEFAULT_GRAPH_CANDIDATE_K, type=int)
    graph_parser.add_argument("--lambda-content", default=DEFAULT_GRAPH_LAMBDA_CONTENT, type=float)
    graph_parser.add_argument("--lambda-behavior", default=DEFAULT_GRAPH_LAMBDA_BEHAVIOR, type=float)
    graph_parser.add_argument("--lambda-temporal", default=DEFAULT_GRAPH_LAMBDA_TEMPORAL, type=float)
    graph_parser.add_argument("--lambda-network", default=DEFAULT_GRAPH_LAMBDA_NETWORK, type=float)
    graph_parser.add_argument(
        "--symmetrize",
        choices=("union_max", "mutual_max", "directed"),
        default=DEFAULT_GRAPH_SYMMETRIZE,
    )
    graph_parser.add_argument("--max-users", required=False, type=int)
    graph_parser.set_defaults(func=run_build_user_graph)

    community_parser = subparsers.add_parser(
        "detect-communities",
        help="Detect user communities from the exported similarity graph",
    )
    community_parser.add_argument("--sample-root", required=True, type=Path)
    community_parser.add_argument("--graph-root", required=False, type=Path)
    community_parser.add_argument("--output-root", required=False, type=Path)
    community_parser.add_argument(
        "--algorithm",
        choices=("structural_entropy", "weighted_lpa"),
        default=DEFAULT_COMMUNITY_ALGORITHM,
    )
    community_parser.add_argument(
        "--max-iterations",
        default=DEFAULT_COMMUNITY_MAX_ITERATIONS,
        type=int,
    )
    community_parser.add_argument(
        "--min-community-size",
        default=DEFAULT_COMMUNITY_MIN_SIZE,
        type=int,
    )
    community_parser.add_argument("--seed", default=config.DEFAULT_SEED, type=int)
    community_parser.add_argument(
        "--mutual-support-bonus",
        default=DEFAULT_COMMUNITY_MUTUAL_SUPPORT_BONUS,
        type=float,
    )
    community_parser.set_defaults(func=run_detect_communities)

    eval_parser = subparsers.add_parser(
        "evaluate-communities",
        help="Evaluate communities by projecting train-split community bot scores to users",
    )
    eval_parser.add_argument("--sample-root", required=True, type=Path)
    eval_parser.add_argument("--communities-root", required=False, type=Path)
    eval_parser.add_argument("--output-root", required=False, type=Path)
    eval_parser.add_argument("--threshold", default=DEFAULT_EVAL_THRESHOLD, type=float)
    eval_parser.add_argument("--smoothing-alpha", default=DEFAULT_EVAL_SMOOTHING_ALPHA, type=float)
    eval_parser.set_defaults(func=run_evaluate_communities)

    purity_parser = subparsers.add_parser(
        "evaluate-community-purity",
        help="Evaluate communities with purity and train-majority label projection",
    )
    purity_parser.add_argument("--sample-root", required=True, type=Path)
    purity_parser.add_argument("--communities-root", required=False, type=Path)
    purity_parser.add_argument("--output-root", required=False, type=Path)
    purity_parser.add_argument("--threshold", default=DEFAULT_PURITY_THRESHOLD, type=float)
    purity_parser.add_argument("--smoothing-alpha", default=DEFAULT_PURITY_SMOOTHING_ALPHA, type=float)
    purity_parser.set_defaults(func=run_evaluate_community_purity)

    structure_parser = subparsers.add_parser(
        "analyze-community-structure",
        help="Analyze structural differences among detected communities",
    )
    structure_parser.add_argument("--sample-root", required=True, type=Path)
    structure_parser.add_argument("--communities-root", required=False, type=Path)
    structure_parser.add_argument("--graph-root", required=False, type=Path)
    structure_parser.add_argument("--purity-root", required=False, type=Path)
    structure_parser.add_argument("--output-root", required=False, type=Path)
    structure_parser.set_defaults(func=run_analyze_community_structure)

    sweep_parser = subparsers.add_parser(
        "sweep-community-pipeline",
        help="Run a parameter sweep over graph construction, community detection, and evaluation",
    )
    sweep_parser.add_argument("--sample-root", required=True, type=Path)
    sweep_parser.add_argument("--vector-root", required=False, type=Path)
    sweep_parser.add_argument("--feature-root", required=False, type=Path)
    sweep_parser.add_argument("--temporal-root", required=False, type=Path)
    sweep_parser.add_argument("--output-root", required=False, type=Path)
    sweep_parser.add_argument("--k-values", nargs="+", default=[5, 10, 15], type=int)
    sweep_parser.add_argument("--min-similarity-values", nargs="+", default=[0.0, 0.05], type=float)
    sweep_parser.add_argument("--min-community-size-values", nargs="+", default=[1, 3, 5], type=int)
    sweep_parser.add_argument("--threshold-values", nargs="+", default=[0.2, 0.3, 0.4, 0.5], type=float)
    sweep_parser.add_argument(
        "--algorithm-values",
        nargs="+",
        default=["structural_entropy", "weighted_lpa"],
    )
    sweep_parser.add_argument(
        "--fusion-mode",
        choices=("early", "late"),
        default=DEFAULT_GRAPH_FUSION_MODE,
    )
    sweep_parser.add_argument("--graph-backend", choices=("auto", "numpy", "python"), default="auto")
    sweep_parser.add_argument("--graph-symmetrize", choices=("union_max", "mutual_max", "directed"), default="union_max")
    sweep_parser.add_argument("--graph-chunk-size", default=512, type=int)
    sweep_parser.add_argument("--candidate-k", default=DEFAULT_GRAPH_CANDIDATE_K, type=int)
    sweep_parser.add_argument("--lambda-content", default=DEFAULT_GRAPH_LAMBDA_CONTENT, type=float)
    sweep_parser.add_argument("--lambda-behavior", default=DEFAULT_GRAPH_LAMBDA_BEHAVIOR, type=float)
    sweep_parser.add_argument("--lambda-temporal", default=DEFAULT_GRAPH_LAMBDA_TEMPORAL, type=float)
    sweep_parser.add_argument("--lambda-network", default=DEFAULT_GRAPH_LAMBDA_NETWORK, type=float)
    sweep_parser.add_argument("--max-iterations", default=DEFAULT_COMMUNITY_MAX_ITERATIONS, type=int)
    sweep_parser.add_argument("--seed", default=config.DEFAULT_SEED, type=int)
    sweep_parser.add_argument("--mutual-support-bonus", default=DEFAULT_COMMUNITY_MUTUAL_SUPPORT_BONUS, type=float)
    sweep_parser.add_argument("--smoothing-alpha", default=DEFAULT_EVAL_SMOOTHING_ALPHA, type=float)
    sweep_parser.add_argument("--objective-split", default=DEFAULT_SWEEP_OBJECTIVE_SPLIT)
    sweep_parser.add_argument("--objective-metric", default=DEFAULT_SWEEP_OBJECTIVE_METRIC)
    sweep_parser.add_argument("--force", action="store_true")
    sweep_parser.set_defaults(func=run_sweep_community_pipeline)

    finalize_parser = subparsers.add_parser(
        "finalize-best-community-run",
        help="Freeze the best sweep run into a canonical output directory",
    )
    finalize_parser.add_argument("--sample-root", required=True, type=Path)
    finalize_parser.add_argument("--sweep-root", required=False, type=Path)
    finalize_parser.add_argument("--output-root", required=False, type=Path)
    finalize_parser.add_argument("--top-communities", default=DEFAULT_FINALIZE_TOP_COMMUNITIES, type=int)
    finalize_parser.set_defaults(func=run_finalize_best_community_run)

    error_parser = subparsers.add_parser(
        "analyze-community-errors",
        help="Generate false-positive/false-negative and community error reports from the finalized best run",
    )
    error_parser.add_argument("--sample-root", required=True, type=Path)
    error_parser.add_argument("--best-root", required=False, type=Path)
    error_parser.add_argument("--output-root", required=False, type=Path)
    error_parser.add_argument("--focus-split", default=DEFAULT_ERROR_ANALYSIS_SPLIT)
    error_parser.add_argument("--top-k", default=DEFAULT_ERROR_ANALYSIS_TOP_K, type=int)
    error_parser.set_defaults(func=run_analyze_community_errors)

    reranker_parser = subparsers.add_parser(
        "train-community-reranker",
        help="Train a second-stage reranker over community scores and user features",
    )
    reranker_parser.add_argument("--sample-root", required=True, type=Path)
    reranker_parser.add_argument("--best-root", required=False, type=Path)
    reranker_parser.add_argument("--output-root", required=False, type=Path)
    reranker_parser.add_argument("--learning-rate", default=DEFAULT_RERANKER_LEARNING_RATE, type=float)
    reranker_parser.add_argument("--max-epochs", default=DEFAULT_RERANKER_MAX_EPOCHS, type=int)
    reranker_parser.add_argument("--l2", default=DEFAULT_RERANKER_L2, type=float)
    reranker_parser.add_argument(
        "--threshold-values",
        nargs="+",
        default=list(DEFAULT_RERANKER_THRESHOLD_VALUES),
        type=float,
    )
    reranker_parser.add_argument(
        "--early-stopping-rounds",
        default=DEFAULT_RERANKER_EARLY_STOPPING,
        type=int,
    )
    reranker_parser.set_defaults(func=run_train_community_reranker)

    reranker_analysis_parser = subparsers.add_parser(
        "analyze-community-reranker",
        help="Compare baseline community predictions against reranker predictions",
    )
    reranker_analysis_parser.add_argument("--sample-root", required=True, type=Path)
    reranker_analysis_parser.add_argument("--best-root", required=False, type=Path)
    reranker_analysis_parser.add_argument("--reranker-root", required=False, type=Path)
    reranker_analysis_parser.add_argument("--output-root", required=False, type=Path)
    reranker_analysis_parser.add_argument("--focus-split", default=DEFAULT_RERANKER_ANALYSIS_SPLIT)
    reranker_analysis_parser.add_argument("--top-k", default=DEFAULT_RERANKER_ANALYSIS_TOP_K, type=int)
    reranker_analysis_parser.set_defaults(func=run_analyze_community_reranker)

    feature_baseline_parser = subparsers.add_parser(
        "run-feature-baselines",
        help="Run Logistic Regression and Random Forest baselines on user features",
    )
    feature_baseline_parser.add_argument("--sample-root", required=True, type=Path)
    feature_baseline_parser.add_argument("--feature-root", required=False, type=Path)
    feature_baseline_parser.add_argument("--output-root", required=False, type=Path)
    feature_baseline_parser.add_argument("--lr-c-values", nargs="+", default=list(DEFAULT_LR_C_VALUES), type=float)
    feature_baseline_parser.add_argument("--class-weight-values", nargs="+", default=["none", "balanced"])
    feature_baseline_parser.add_argument("--rf-estimators", nargs="+", default=list(DEFAULT_RF_ESTIMATORS), type=int)
    feature_baseline_parser.add_argument("--rf-max-depths", nargs="+", default=["8", "16", "none"])
    feature_baseline_parser.add_argument("--seed", default=DEFAULT_BASELINE_SEED, type=int)
    feature_baseline_parser.set_defaults(func=run_external_feature_baselines)

    graph_baseline_parser = subparsers.add_parser(
        "run-graph-baselines",
        help="Run DeepWalk/Node2Vec baselines on the original following graph",
    )
    graph_baseline_parser.add_argument("--sample-root", required=True, type=Path)
    graph_baseline_parser.add_argument("--output-root", required=False, type=Path)
    graph_baseline_parser.add_argument("--dimension", default=DEFAULT_WALK_DIMENSION, type=int)
    graph_baseline_parser.add_argument("--walk-length", default=DEFAULT_WALK_LENGTH, type=int)
    graph_baseline_parser.add_argument("--num-walks", default=DEFAULT_NUM_WALKS, type=int)
    graph_baseline_parser.add_argument("--window", default=DEFAULT_WALK_WINDOW, type=int)
    graph_baseline_parser.add_argument("--epochs", default=DEFAULT_WALK_EPOCHS, type=int)
    graph_baseline_parser.add_argument("--lr-c-values", nargs="+", default=list(DEFAULT_LR_C_VALUES), type=float)
    graph_baseline_parser.add_argument("--class-weight-values", nargs="+", default=["none", "balanced"])
    graph_baseline_parser.add_argument("--node2vec-p-values", nargs="+", default=list(DEFAULT_NODE2VEC_P_VALUES), type=float)
    graph_baseline_parser.add_argument("--node2vec-q-values", nargs="+", default=list(DEFAULT_NODE2VEC_Q_VALUES), type=float)
    graph_baseline_parser.add_argument("--seed", default=DEFAULT_BASELINE_SEED, type=int)
    graph_baseline_parser.set_defaults(func=run_external_graph_baselines)

    kmeans_parser = subparsers.add_parser(
        "run-kmeans-baseline",
        help="Run K-Means grouping baseline with purity-based label projection",
    )
    kmeans_parser.add_argument("--sample-root", required=True, type=Path)
    kmeans_parser.add_argument("--feature-root", required=False, type=Path)
    kmeans_parser.add_argument("--output-root", required=False, type=Path)
    kmeans_parser.add_argument("--k-values", nargs="+", default=list(DEFAULT_KMEANS_K_VALUES), type=int)
    kmeans_parser.add_argument("--threshold", default=DEFAULT_PURITY_THRESHOLD, type=float)
    kmeans_parser.add_argument("--smoothing-alpha", default=DEFAULT_PURITY_SMOOTHING_ALPHA, type=float)
    kmeans_parser.add_argument("--seed", default=DEFAULT_BASELINE_SEED, type=int)
    kmeans_parser.set_defaults(func=run_kmeans_baseline)

    grouping_summary_parser = subparsers.add_parser(
        "summarize-grouping-baselines",
        help="Aggregate K-Means, Weighted LPA, and Structural Entropy grouping baselines",
    )
    grouping_summary_parser.add_argument("--sample-root", required=True, type=Path)
    grouping_summary_parser.add_argument("--output-root", required=False, type=Path)
    grouping_summary_parser.add_argument("--kmeans-root", required=False, type=Path)
    grouping_summary_parser.add_argument("--weighted-lpa-purity-root", required=False, type=Path)
    grouping_summary_parser.add_argument("--structural-entropy-purity-root", required=False, type=Path)
    grouping_summary_parser.set_defaults(func=run_summarize_grouping_baselines)

    summary_baseline_parser = subparsers.add_parser(
        "summarize-external-baselines",
        help="Aggregate completed external baselines into a paper-ready summary table",
    )
    summary_baseline_parser.add_argument("--sample-root", required=True, type=Path)
    summary_baseline_parser.add_argument("--baselines-root", required=False, type=Path)
    summary_baseline_parser.add_argument("--output-root", required=False, type=Path)
    summary_baseline_parser.set_defaults(func=run_summarize_external_baselines)
    return parser


def run_profile(args: argparse.Namespace) -> None:
    output_path = build_user_profile(args.data_root, args.work_root)
    print(output_path)


def run_sample(args: argparse.Namespace) -> None:
    thresholds = config.SamplingThresholds(
        max_context_mutual=args.max_context_mutual,
        max_context_follower=args.max_context_follower,
        max_context_following=args.max_context_following,
        seed_user_max_tweets=args.seed_user_max_tweets,
        context_user_max_tweets=args.context_user_max_tweets,
    )

    profile_path = args.work_root / "profile" / "users_profile.csv"
    if not profile_path.exists():
        profile_path = build_user_profile(args.data_root, args.work_root)
    profile_rows = load_profile_rows(profile_path)
    profile_by_user_id = {row["user_id"]: row for row in profile_rows}

    split_map = read_split_map(resolve_split_path(args.data_root))
    label_map = read_label_map(resolve_label_path(args.data_root))
    edge_path = resolve_edge_path(args.data_root)
    tweet_paths = resolve_tweet_paths(args.data_root)

    seed_user_ids, seed_sampling_summary = select_seed_users(
        profile_rows,
        preset=args.preset,
        seed=args.seed,
    )
    final_user_ids, context_summary = expand_context_users(
        edge_path,
        seed_user_ids=seed_user_ids,
        profile_by_user_id=profile_by_user_id,
        thresholds=thresholds,
        seed=args.seed,
    )

    post_candidates_by_user, user_edges, second_pass_summary = collect_post_candidates_and_user_edges(
        edge_path,
        final_user_ids=final_user_ids,
        seed_user_ids=set(seed_user_ids),
        thresholds=thresholds,
        seed=args.seed,
    )
    candidate_tweet_ids = {tweet_id for tweet_ids in post_candidates_by_user.values() for tweet_id in tweet_ids}
    candidate_tweet_records = load_tweet_records(tweet_paths, candidate_tweet_ids)

    selected_tweet_records, post_edges, post_selection_summary = finalize_post_selection(
        post_candidates_by_user,
        tweet_records=candidate_tweet_records,
        seed_user_ids=set(seed_user_ids),
        thresholds=thresholds,
        seed=args.seed,
    )
    final_tweet_records, reference_summary = expand_referenced_tweets(
        tweet_paths,
        selected_tweet_records=selected_tweet_records,
    )
    extra_edges, tweet_relation_counts = collect_third_pass_edges(
        edge_path,
        final_user_ids=final_user_ids,
        final_tweet_ids=set(final_tweet_records),
    )

    user_records = collect_user_records(args.data_root, final_user_ids)
    existing_user_ids = set(user_records)
    existing_tweet_ids = set(final_tweet_records)
    filtered_user_edges = {
        edge for edge in user_edges if edge[0] in existing_user_ids and edge[1] in existing_user_ids
    }
    filtered_post_edges = {
        edge for edge in post_edges if edge[0] in existing_user_ids and edge[1] in existing_tweet_ids
    }
    filtered_extra_edges = {
        edge
        for edge in extra_edges
        if edge[0] in existing_tweet_ids
        and ((edge[2] == "mention" and edge[1] in existing_user_ids) or (edge[2] != "mention" and edge[1] in existing_tweet_ids))
    }

    manifest = export_sample_dataset(
        args.output_root,
        preset=args.preset,
        seed=args.seed,
        source_data_root=args.data_root,
        user_records=user_records,
        tweet_records=final_tweet_records,
        user_edges=filtered_user_edges,
        post_edges=filtered_post_edges,
        extra_edges=filtered_extra_edges,
        split_map=split_map,
        label_map=label_map,
        profile_by_user_id=profile_by_user_id,
        seed_sampling_summary=seed_sampling_summary,
        context_summary=context_summary,
        second_pass_summary=second_pass_summary,
        post_selection_summary=post_selection_summary,
        reference_summary=reference_summary,
        tweet_relation_counts=tweet_relation_counts,
        thresholds=thresholds,
    )
    print(args.output_root / "sample_manifest.json")
    print(manifest["final_counts"])


def run_validate(args: argparse.Namespace) -> None:
    result = validate_sample(args.sample_root, args.report_out)
    print(result)


def run_audit(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "analysis" / "field_audit")
    summary = run_field_audit(
        args.sample_root,
        output_root,
        min_triplet_tweets=args.min_triplet_tweets,
        min_time_tweets=args.min_time_tweets,
        min_behavior_tweets=args.min_behavior_tweets,
    )
    print(output_root / "audit_summary.json")
    print(summary["overall"])


def run_extract_triplets(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "derived" / "triplets")
    client = OpenAICompatibleClient(_build_llm_settings(args))
    manifest = run_triplet_extraction(
        args.sample_root,
        output_root,
        client=client,
        seed=args.seed,
        per_user_limit=args.per_user_limit,
        min_user_tweets=args.min_user_tweets,
        max_users=args.max_users,
        max_tweets=args.max_tweets,
        overwrite=args.overwrite,
    )
    print(output_root / "run_manifest.json")
    print(
        {
            "processed_count": manifest["processed_count"],
            "skipped_count": manifest["skipped_count"],
            "error_count": manifest["error_count"],
            "user_document_count": manifest["user_document_count"],
        }
    )


def run_classify_post_types(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "derived" / "post_types")
    client = None
    if args.mode in {"hybrid", "llm"}:
        client = OpenAICompatibleClient(_build_llm_settings(args))
    manifest = run_post_type_classification(
        args.sample_root,
        output_root,
        mode=args.mode,
        client=client,
        seed=args.seed,
        per_user_limit=args.per_user_limit,
        min_user_tweets=args.min_user_tweets,
        max_users=args.max_users,
        max_tweets=args.max_tweets,
        overwrite=args.overwrite,
    )
    print(output_root / "run_manifest.json")
    print(
        {
            "heuristic_count": manifest["heuristic_count"],
            "llm_count": manifest["llm_count"],
            "skipped_count": manifest["skipped_count"],
            "error_count": manifest["error_count"],
            "user_distribution_count": manifest["user_distribution_count"],
        }
    )


def run_downsample_final(args: argparse.Namespace) -> None:
    manifest = downsample_exported_sample(
        args.sample_root,
        args.output_root,
        target_users=args.target_users,
        seed=args.seed,
    )
    print(args.output_root / "sample_manifest.json")
    print(manifest["final_counts"])


def run_build_user_features(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "analysis" / "user_features")
    manifest = build_user_feature_table(
        args.sample_root,
        output_root,
        triplet_seed=args.triplet_seed,
        post_type_seed=args.post_type_seed,
    )
    print(output_root / "feature_table_manifest.json")
    print(manifest["counts"])


def run_build_user_vectors(args: argparse.Namespace) -> None:
    feature_root = args.feature_root or (args.sample_root / "analysis" / "user_features")
    output_root = args.output_root or (args.sample_root / "analysis" / "user_vectors")
    client = OpenAICompatibleClient(
        load_llm_settings(
            model=args.embedding_model,
            model_env="OPENAI_EMBEDDING_MODEL",
            default_model=DEFAULT_EMBEDDING_MODEL,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            env_file=args.env_file,
            env_search_roots=(args.sample_root,),
            timeout_s=args.request_timeout_s,
            max_retries=args.max_retries,
            temperature=0.0,
            requests_per_minute=args.requests_per_minute,
        )
    )
    manifest = build_user_vectors(
        feature_root,
        output_root,
        client=client,
        text_field=args.text_field,
        fallback_to_description=not args.no_description_fallback,
        batch_size=args.batch_size,
        max_users=args.max_users,
    )
    print(output_root / "vector_manifest.json")
    print(
        {
            "users": manifest["counts"]["users"],
            "embedded_users": manifest["counts"]["embedded_users"],
            "missing_text_users": manifest["counts"]["missing_text_users"],
            "embedding_dim": manifest["embedding_dim"],
            "numeric_dim": manifest["numeric_dim"],
            "fused_dim": manifest["fused_dim"],
        }
    )


def run_build_temporal_profiles(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "analysis" / "temporal_profiles")
    manifest = build_temporal_profiles(
        args.sample_root,
        output_root,
        min_time_tweets=args.min_time_tweets,
    )
    print(output_root / "temporal_manifest.json")
    print(
        {
            "users": manifest["counts"]["users"],
            "temporal_ready_users": manifest["counts"]["temporal_ready_users"],
            "tweets_with_created_at": manifest["counts"]["tweets_with_created_at"],
        }
    )


def run_build_user_graph(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "analysis" / "user_graph")
    manifest = build_user_graph(
        args.sample_root if args.fusion_mode == "late" else (args.vector_root or (args.sample_root / "analysis" / "user_vectors")),
        output_root,
        k=args.k,
        metric=args.metric,
        min_similarity=args.min_similarity,
        backend=args.backend,
        symmetrize=args.symmetrize,
        chunk_size=args.chunk_size,
        max_users=args.max_users,
        fusion_mode=args.fusion_mode,
        vector_root=args.vector_root,
        feature_root=args.feature_root,
        temporal_root=args.temporal_root,
        candidate_k=args.candidate_k,
        lambda_content=args.lambda_content,
        lambda_behavior=args.lambda_behavior,
        lambda_temporal=args.lambda_temporal,
        lambda_network=args.lambda_network,
    )
    print(output_root / "graph_manifest.json")
    payload = {
        "fusion_mode": manifest["fusion_mode"],
        "users": manifest["counts"]["users"],
        "directed_edges": manifest["counts"]["directed_edges"],
        "undirected_edges": manifest["counts"]["undirected_edges"],
        "backend": manifest["backend"],
        "k": manifest["k"],
    }
    if manifest["fusion_mode"] == "late":
        payload["candidate_k"] = manifest["candidate_k"]
    else:
        payload["vector_dim"] = manifest["vector_dim"]
    print(payload)


def run_detect_communities(args: argparse.Namespace) -> None:
    graph_root = args.graph_root or (args.sample_root / "analysis" / "user_graph")
    output_root = args.output_root or (args.sample_root / "analysis" / "communities")
    manifest = detect_communities(
        args.sample_root,
        graph_root,
        output_root,
        algorithm=args.algorithm,
        max_iterations=args.max_iterations,
        min_community_size=args.min_community_size,
        seed=args.seed,
        mutual_support_bonus=args.mutual_support_bonus,
    )
    print(output_root / "community_manifest.json")
    print(
        {
            "algorithm": manifest["algorithm"],
            "users": manifest["counts"]["users"],
            "communities": manifest["counts"]["communities"],
            "singleton_communities": manifest["counts"]["singleton_communities"],
            "iterations_run": manifest["iterations_run"],
            "merge_count": manifest["merge_count"],
            "largest_community": manifest["size_summary"]["largest_community"],
        }
    )


def run_evaluate_communities(args: argparse.Namespace) -> None:
    communities_root = args.communities_root or (args.sample_root / "analysis" / "communities")
    output_root = args.output_root or (args.sample_root / "analysis" / "community_eval")
    manifest = evaluate_communities(
        args.sample_root,
        communities_root,
        output_root,
        threshold=args.threshold,
        smoothing_alpha=args.smoothing_alpha,
    )
    print(output_root / "community_eval_manifest.json")
    print(
        {
            "users": manifest["counts"]["users"],
            "communities": manifest["counts"]["communities"],
            "labeled_users": manifest["counts"]["labeled_users"],
            "test_f1": manifest["metrics"].get("test", {}).get("f1", 0.0),
            "test_auc": manifest["metrics"].get("test", {}).get("auc", 0.0),
        }
    )


def run_evaluate_community_purity(args: argparse.Namespace) -> None:
    communities_root = args.communities_root or (args.sample_root / "analysis" / "communities")
    output_root = args.output_root or (args.sample_root / "analysis" / "community_purity")
    manifest = evaluate_community_purity(
        args.sample_root,
        communities_root,
        output_root,
        threshold=args.threshold,
        smoothing_alpha=args.smoothing_alpha,
    )
    print(output_root / "community_purity_manifest.json")
    print(
        {
            "method": manifest["method_name"],
            "communities": manifest["counts"]["communities"],
            "global_purity": manifest["global_purity"],
            "test_f1": manifest["metrics"].get("test", {}).get("f1", 0.0),
            "test_auc": manifest["metrics"].get("test", {}).get("auc", 0.0),
        }
    )


def run_analyze_community_structure(args: argparse.Namespace) -> None:
    communities_root = args.communities_root or (args.sample_root / "analysis" / "communities")
    graph_root = args.graph_root or (args.sample_root / "analysis" / "user_graph")
    purity_root = args.purity_root or (args.sample_root / "analysis" / "community_purity")
    output_root = args.output_root or (args.sample_root / "analysis" / "community_structure")
    manifest = analyze_community_structure(
        args.sample_root,
        communities_root,
        graph_root,
        purity_root,
        output_root,
    )
    print(output_root / "community_structure_manifest.json")
    print(
        {
            "communities": manifest["counts"]["communities"],
            "representative_rows": manifest["counts"]["representative_rows"],
            "archetypes": manifest["archetype_counts"],
        }
    )


def run_sweep_community_pipeline(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "analysis" / "community_sweep")
    manifest = sweep_community_pipeline(
        args.sample_root,
        output_root,
        vector_root=args.vector_root,
        feature_root=args.feature_root,
        temporal_root=args.temporal_root,
        k_values=args.k_values,
        min_similarity_values=args.min_similarity_values,
        min_community_size_values=args.min_community_size_values,
        threshold_values=args.threshold_values,
        algorithm_values=args.algorithm_values,
        fusion_mode=args.fusion_mode,
        graph_backend=args.graph_backend,
        graph_symmetrize=args.graph_symmetrize,
        graph_chunk_size=args.graph_chunk_size,
        candidate_k=args.candidate_k,
        lambda_content=args.lambda_content,
        lambda_behavior=args.lambda_behavior,
        lambda_temporal=args.lambda_temporal,
        lambda_network=args.lambda_network,
        max_iterations=args.max_iterations,
        seed=args.seed,
        mutual_support_bonus=args.mutual_support_bonus,
        smoothing_alpha=args.smoothing_alpha,
        objective_split=args.objective_split,
        objective_metric=args.objective_metric,
        force=args.force,
    )
    print(output_root / "community_sweep_manifest.json")
    best_run = manifest.get("best_run", {})
    print(
        {
            "run_count": manifest["run_count"],
            "best_run": best_run.get("run_name", ""),
            "objective_split": manifest["objective"]["split"],
            "test_f1": best_run.get("test_f1", 0.0),
            "test_auc": best_run.get("test_auc", 0.0),
        }
    )


def run_finalize_best_community_run(args: argparse.Namespace) -> None:
    sweep_root = args.sweep_root or (args.sample_root / "analysis" / "community_sweep")
    output_root = args.output_root or (args.sample_root / "analysis" / "community_best")
    manifest = finalize_best_community_run(
        sweep_root,
        output_root,
        top_communities=args.top_communities,
    )
    selected_run = manifest["selected_run"]
    print(output_root / "best_run_manifest.json")
    print(
        {
            "run_name": selected_run.get("run_name", ""),
            "test_f1": selected_run.get("test_f1", 0.0),
            "test_auc": selected_run.get("test_auc", 0.0),
            "output_root": str(output_root),
        }
    )


def run_analyze_community_errors(args: argparse.Namespace) -> None:
    best_root = args.best_root or (args.sample_root / "analysis" / "community_best")
    output_root = args.output_root or (args.sample_root / "analysis" / "community_error_analysis")
    manifest = analyze_community_errors(
        args.sample_root,
        best_root,
        output_root,
        focus_split=args.focus_split,
        top_k=args.top_k,
    )
    print(output_root / "error_analysis_manifest.json")
    print(
        {
            "focus_split": manifest["focus_split"],
            "false_positives_focus": manifest["counts"]["false_positives_focus"],
            "false_negatives_focus": manifest["counts"]["false_negatives_focus"],
            "communities": manifest["counts"]["communities"],
        }
    )


def run_train_community_reranker(args: argparse.Namespace) -> None:
    best_root = args.best_root or (args.sample_root / "analysis" / "community_best")
    output_root = args.output_root or (args.sample_root / "analysis" / "community_reranker")
    manifest = train_community_reranker(
        args.sample_root,
        best_root,
        output_root,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        l2=args.l2,
        threshold_values=args.threshold_values,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(output_root / "reranker_manifest.json")
    print(
        {
            "selected_threshold": manifest["training"]["selected_threshold"],
            "baseline_test_f1": manifest["baseline_metrics"].get("test", {}).get("f1", 0.0),
            "reranker_test_f1": manifest["reranker_metrics"].get("test", {}).get("f1", 0.0),
            "baseline_test_auc": manifest["baseline_metrics"].get("test", {}).get("auc", 0.0),
            "reranker_test_auc": manifest["reranker_metrics"].get("test", {}).get("auc", 0.0),
        }
    )


def run_analyze_community_reranker(args: argparse.Namespace) -> None:
    best_root = args.best_root or (args.sample_root / "analysis" / "community_best")
    reranker_root = args.reranker_root or (args.sample_root / "analysis" / "community_reranker")
    output_root = args.output_root or (args.sample_root / "analysis" / "community_reranker_analysis")
    manifest = analyze_community_reranker(
        args.sample_root,
        best_root,
        reranker_root,
        output_root,
        focus_split=args.focus_split,
        top_k=args.top_k,
    )
    print(output_root / "reranker_comparison_manifest.json")
    print(
        {
            "focus_split": manifest["focus_split"],
            "changed_predictions": manifest["counts"]["changed_predictions"],
            "fixed_cases": manifest["counts"]["fixed_cases"],
            "regressed_cases": manifest["counts"]["regressed_cases"],
        }
    )


def run_external_feature_baselines(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "analysis" / "external_baselines_10k")
    manifests = run_feature_baselines(
        args.sample_root,
        output_root,
        feature_root=args.feature_root,
        lr_c_values=tuple(float(value) for value in args.lr_c_values),
        class_weight_values=_normalize_class_weight_values(args.class_weight_values),
        rf_estimators=tuple(int(value) for value in args.rf_estimators),
        rf_max_depths=tuple(_parse_optional_int(value) for value in args.rf_max_depths),
        seed=args.seed,
    )
    print(output_root)
    print({key: manifest["selected_params"] for key, manifest in manifests.items()})


def run_external_graph_baselines(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "analysis" / "external_baselines_10k")
    manifests = run_graph_baselines(
        args.sample_root,
        output_root,
        dimension=args.dimension,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window=args.window,
        epochs=args.epochs,
        lr_c_values=tuple(float(value) for value in args.lr_c_values),
        class_weight_values=_normalize_class_weight_values(args.class_weight_values),
        node2vec_p_values=tuple(float(value) for value in args.node2vec_p_values),
        node2vec_q_values=tuple(float(value) for value in args.node2vec_q_values),
        seed=args.seed,
    )
    print(output_root)
    print({key: manifest["selected_params"] for key, manifest in manifests.items()})


def run_kmeans_baseline(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "analysis" / "grouping_baselines_10k" / "kmeans")
    manifest = run_kmeans_grouping_baseline(
        args.sample_root,
        output_root,
        feature_root=args.feature_root,
        k_values=tuple(int(value) for value in args.k_values),
        threshold=args.threshold,
        smoothing_alpha=args.smoothing_alpha,
        seed=args.seed,
    )
    print(output_root / "community_purity_manifest.json")
    print(
        {
            "selected_k": manifest["selected_params"].get("n_clusters", 0),
            "communities": manifest["counts"]["communities"],
            "global_purity": manifest["global_purity"],
            "test_f1": manifest["metrics"].get("test", {}).get("f1", 0.0),
            "test_auc": manifest["metrics"].get("test", {}).get("auc", 0.0),
        }
    )


def run_summarize_grouping_baselines(args: argparse.Namespace) -> None:
    output_root = args.output_root or (args.sample_root / "analysis" / "grouping_baselines_10k" / "summary")
    manifest = summarize_grouping_baselines(
        args.sample_root,
        output_root,
        kmeans_root=args.kmeans_root or (args.sample_root / "analysis" / "grouping_baselines_10k" / "kmeans"),
        weighted_lpa_purity_root=args.weighted_lpa_purity_root
        or (args.sample_root / "analysis" / "grouping_baselines_10k" / "weighted_lpa" / "community_purity"),
        structural_entropy_purity_root=args.structural_entropy_purity_root
        or (args.sample_root / "analysis" / "run_10k_late" / "community_purity"),
    )
    print(output_root / "grouping_baseline_manifest.json")
    print(manifest["counts"])


def run_summarize_external_baselines(args: argparse.Namespace) -> None:
    baselines_root = args.baselines_root or (args.sample_root / "analysis" / "external_baselines_10k")
    output_root = args.output_root or (baselines_root / "summary")
    manifest = summarize_external_baselines(
        args.sample_root,
        baselines_root,
        output_root,
    )
    print(output_root / "external_baseline_manifest.json")
    print(manifest["counts"])


def _normalize_class_weight_values(values: list[str]) -> tuple[str | None, ...]:
    normalized: list[str | None] = []
    for value in values:
        lowered = str(value).strip().lower()
        if lowered in {"none", "", "null"}:
            normalized.append(None)
        else:
            normalized.append(lowered)
    return tuple(normalized)


def _parse_optional_int(value: str) -> int | None:
    lowered = str(value).strip().lower()
    if lowered in {"none", "", "null"}:
        return None
    return int(lowered)


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=False)
    parser.add_argument("--base-url", required=False)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", required=False, type=Path)
    parser.add_argument("--concurrency", required=False, type=int)
    parser.add_argument("--requests-per-minute", required=False, type=int)
    parser.add_argument("--request-timeout-s", default=60.0, type=float)
    parser.add_argument("--max-retries", default=3, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)


def _build_llm_settings(args: argparse.Namespace):
    return load_llm_settings(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        env_file=args.env_file,
        env_search_roots=(args.sample_root,),
        timeout_s=args.request_timeout_s,
        max_retries=args.max_retries,
        temperature=args.temperature,
        concurrency=args.concurrency,
        requests_per_minute=args.requests_per_minute,
    )


if __name__ == "__main__":
    main()

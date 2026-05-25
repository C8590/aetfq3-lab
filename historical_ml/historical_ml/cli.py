from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from .audit import generate_replay_audit_report
from .calibration import generate_entry_calibration_outputs
from .config import HistoricalMLConfig
from .io_utils import read_price_data, read_table, write_table
from .labeler import FutureLabeler, MLEntryUniverseLabeler
from .ml_baseline import run_baseline_from_file, run_baseline
from .ml_entry_model import score_entry_quality_universe, score_entry_quality_universe_from_files, train_entry_quality_model_from_file
from .ml_core_recovered_review import build_ml_core_recovered_review_from_file
from .ml_recovered_thresholds import build_ml_recovered_threshold_recommendation_from_file
from .ml_sim_historical_review import build_ml_sim_historical_review_from_files
from .ml_sim_review import build_ml_sim_weekly_review_from_files
from .ml_stability import run_ml_stability_from_file
from .performance import empty_timing, write_performance_outputs
from .policy_comparison import compare_ml_policies_from_files
from .recommendations import make_recommendations_from_artifacts
from .replay_engine import HistoricalReplayEngine
from .reports import generate_entry_threshold_report
from .review_queue import build_manual_review_queue
from .coverage import generate_ml_universe_coverage_report
from .sector_mapping_audit import build_sector_mapping_audit_from_files
from .sector_mapping_suggestion import (
    apply_sector_mapping_top100_from_files,
    build_sector_mapping_suggestions_from_files,
    build_sector_mapping_top100_review_from_files,
)


def _trim_prices_by_trading_window(prices, start, end, *, before: int = 0, after: int = 0):
    if prices.empty:
        return prices
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    dates = list(frame["date"].dropna().sort_values().unique())
    if not dates:
        return frame
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    start_idx = next((idx for idx, value in enumerate(dates) if value >= start_ts), 0)
    end_idx = max((idx for idx, value in enumerate(dates) if value <= end_ts), default=len(dates) - 1)
    lo = max(0, start_idx - before)
    hi = min(len(dates) - 1, end_idx + after)
    return frame.loc[(frame["date"] >= dates[lo]) & (frame["date"] <= dates[hi])].copy()


def _read_scoring_samples(path: str | Path, score_date: str | None):
    sample_path = Path(path)
    if not score_date or sample_path.suffix.lower() in {".parquet", ".pq"}:
        return read_table(sample_path)
    target = pd.Timestamp(score_date).strftime("%Y-%m-%d")
    chunks = []
    for chunk in pd.read_csv(sample_path, chunksize=100_000):
        if "trade_date" not in chunk.columns:
            chunks.append(chunk)
            continue
        mask = chunk["trade_date"].astype(str).str.slice(0, 10).eq(target)
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="historical_ml sample factory")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--prices", required=True, help="ETF daily price CSV/parquet: date,code,name,close,sector,...")
        sp.add_argument("--start", default="2024-09-24")
        sp.add_argument("--end", default="2026-05-19")
        sp.add_argument("--out", required=True)
        sp.add_argument("--format", choices=["csv", "parquet"], default="csv")
        sp.add_argument("--market-code", default=None)

    replay = sub.add_parser("replay", help="produce daily feature/candidate samples without future labels")
    add_common(replay)

    label = sub.add_parser("label", help="attach future labels to entry_candidate_samples")
    label.add_argument("--prices", required=True)
    label.add_argument("--samples", required=True)
    label.add_argument("--out", required=True)
    label.add_argument("--format", choices=["csv", "parquet"], default="csv")
    label.add_argument("--market-code", default=None)

    universe_label = sub.add_parser("label-universe", help="attach training labels to daily_ml_universe_samples")
    universe_label.add_argument("--prices", required=True)
    universe_label.add_argument("--samples", required=True)
    universe_label.add_argument("--out", required=True)
    universe_label.add_argument("--format", choices=["csv", "parquet"], default="csv")
    universe_label.add_argument("--market-code", default=None)
    universe_label.add_argument("--horizons", default="5,10,20", help="comma-separated label horizons, e.g. 5,10,20")

    report = sub.add_parser("report", help="generate manual review queue and entry threshold report")
    report.add_argument("--labeled-samples", required=True)
    report.add_argument("--daily-etf-samples", default=None)
    report.add_argument("--daily-sector-samples", default=None)
    report.add_argument("--daily-decision-snapshot", default=None)
    report.add_argument("--unlabeled-samples", default=None)
    report.add_argument("--out", required=True)
    report.add_argument("--format", choices=["csv", "parquet"], default="csv")

    run_all = sub.add_parser("run-all", help="replay + label + review queue + threshold report")
    add_common(run_all)
    run_all.add_argument("--with-ml", action="store_true", help="also run offline baseline ML diagnostics after labeling")

    train = sub.add_parser("train-baseline", help="train offline baseline ML diagnostic models")
    train.add_argument("--samples", required=True, help="entry_candidate_samples_labeled CSV/parquet")
    train.add_argument("--out", required=True)
    train.add_argument("--target", choices=["both", "good_entry", "bad_entry"], default="both")
    train.add_argument("--format", choices=["csv", "parquet"], default="csv")

    entry_model = sub.add_parser("train-entry-model", help="train shadow ETF code-level entry quality scores")
    entry_model.add_argument("--samples", required=True, help="ml_entry_labeled_samples CSV/parquet")
    entry_model.add_argument("--out", required=True)
    entry_model.add_argument("--format", choices=["csv", "parquet"], default="csv")
    entry_model.add_argument("--top-n", type=int, default=5)
    entry_model.add_argument("--min-train-dates", type=int, default=5)

    score_entry = sub.add_parser("score-entry-universe", help="score current feature-ready ETF universe with the shadow entry model")
    score_entry.add_argument("--labeled-samples", required=True, help="ml_entry_labeled_samples CSV/parquet")
    score_entry.add_argument("--scoring-samples", required=True, help="daily_ml_universe_samples CSV/parquet")
    score_entry.add_argument("--out", required=True)
    score_entry.add_argument("--format", choices=["csv", "parquet"], default="csv")
    score_entry.add_argument("--score-date", default=None)
    score_entry.add_argument("--min-train-dates", type=int, default=5)

    coverage = sub.add_parser("coverage-report", help="diagnose current ML universe and score coverage")
    coverage.add_argument("--prices", required=True)
    coverage.add_argument("--pre-selection", required=True)
    coverage.add_argument("--scores", required=True)
    coverage.add_argument("--daily-universe", required=True)
    coverage.add_argument("--entry-signal", default=None)
    coverage.add_argument("--order-intent", default=None)
    coverage.add_argument("--out", required=True)
    coverage.add_argument("--trade-date", default=None)

    policy_compare = sub.add_parser("compare-policies", help="compare legacy, shadow, candidate expansion, and active_sim policies")
    policy_compare.add_argument("--samples", required=True, help="ml_entry_labeled_samples CSV/parquet")
    policy_compare.add_argument("--scores", required=True, help="walk-forward ml_entry_scores CSV/parquet")
    policy_compare.add_argument("--out", required=True)
    policy_compare.add_argument("--format", choices=["csv", "parquet"], default="csv")

    ml_sim_review = sub.add_parser("ml-sim-review", help="backfill ML_SIM review queue and build weekly learning report")
    ml_sim_review.add_argument("--review-queue", required=True)
    ml_sim_review.add_argument("--prices", required=True)
    ml_sim_review.add_argument("--out", required=True)
    ml_sim_review.add_argument("--comparison", default=None)
    ml_sim_review.add_argument("--summary", default=None)
    ml_sim_review.add_argument("--daily-decision-snapshot", default=None)

    ml_sim_historical = sub.add_parser("ml-sim-historical-review", help="walk-forward historical V2.1_ML_SIM effectiveness review")
    ml_sim_historical.add_argument("--labeled-samples", required=True, help="ml_entry_labeled_samples CSV/parquet")
    ml_sim_historical.add_argument("--scoring-samples", required=True, help="daily_ml_universe_samples CSV/parquet")
    ml_sim_historical.add_argument("--prices", required=True)
    ml_sim_historical.add_argument("--out", required=True)
    ml_sim_historical.add_argument("--start", default="2024-09-24")
    ml_sim_historical.add_argument("--end", default=None)
    ml_sim_historical.add_argument("--recent-days", type=int, default=None, help="optional smoke window over the most recent N labelable trade dates")
    ml_sim_historical.add_argument("--min-train-dates", type=int, default=20)

    recovered_threshold = sub.add_parser("ml-recovered-thresholds", help="offline threshold grid for narrower ML_RECOVERED candidates")
    recovered_threshold.add_argument("--historical-review", required=True, help="ml_sim_historical_review_filled CSV/parquet")
    recovered_threshold.add_argument("--out", required=True)

    core_recovered = sub.add_parser("ml-core-recovered-review", help="offline manual review and ML_CORE_RECOVERED threshold grid")
    core_recovered.add_argument("--historical-review", required=True, help="ml_sim_historical_review_filled CSV/parquet")
    core_recovered.add_argument("--out", required=True)

    sector_audit = sub.add_parser("sector-mapping-audit", help="offline sector mapping audit for ML_STRONG_RECOVERED")
    sector_audit.add_argument("--historical-review", required=True, help="ml_sim_historical_review_filled CSV/parquet")
    sector_audit.add_argument("--out", required=True)
    sector_audit.add_argument("--sector-map", default=str(Path("config") / "etf_sector_map.yaml"))
    sector_audit.add_argument("--universe", default=str(Path("output") / "etf_universe_snapshot.csv"))
    sector_audit.add_argument("--prices", default=str(Path("data") / "etf_daily.csv"))

    sector_suggestion = sub.add_parser("sector-mapping-suggestion", help="offline full-universe ETF sector mapping suggestions")
    sector_suggestion.add_argument("--out", required=True)
    sector_suggestion.add_argument("--sector-map", default=str(Path("config") / "etf_sector_map.yaml"))
    sector_suggestion.add_argument("--universe", default=str(Path("output") / "etf_universe_snapshot.csv"))
    sector_suggestion.add_argument("--prices", default=str(Path("data") / "etf_daily.csv"))
    sector_suggestion.add_argument("--strong-audit", default=str(Path("output") / "ml_strong_recovered_sector_audit.csv"))
    sector_suggestion.add_argument("--historical-review", default=str(Path("output") / "ml_sim_historical_review_filled.csv"))
    sector_suggestion.add_argument("--current-position", default=str(Path("config") / "current_position.yaml"))
    sector_suggestion.add_argument("--entry-signal", default=str(Path("output") / "entry_signal.csv"))

    top100_review = sub.add_parser("sector-mapping-top100-review", help="offline top100 ETF sector mapping manual review package")
    top100_review.add_argument("--out", required=True)
    top100_review.add_argument("--priority-review", default=str(Path("output") / "sector_mapping_priority_review.csv"))
    top100_review.add_argument("--sector-map", default=str(Path("config") / "etf_sector_map.yaml"))
    top100_review.add_argument("--universe", default=str(Path("output") / "etf_universe_snapshot.csv"))
    top100_review.add_argument("--prices", default=str(Path("data") / "etf_daily.csv"))
    top100_review.add_argument("--strong-audit", default=str(Path("output") / "ml_strong_recovered_sector_audit.csv"))
    top100_review.add_argument("--historical-review", default=str(Path("output") / "ml_sim_historical_review_filled.csv"))
    top100_review.add_argument("--entry-signal", default=str(Path("output") / "entry_signal.csv"))
    top100_review.add_argument("--top-n", type=int, default=100)

    apply_top100 = sub.add_parser("sector-mapping-apply-top100", help="append high-confidence top100 sector map rows and audit after apply")
    apply_top100.add_argument("--out", required=True)
    apply_top100.add_argument("--sector-map", default=str(Path("config") / "etf_sector_map.yaml"))
    apply_top100.add_argument("--review", default=str(Path("output") / "sector_mapping_top100_review.csv"))
    apply_top100.add_argument("--patch-draft", default=str(Path("output") / "sector_mapping_top100_patch_draft.yaml"))
    apply_top100.add_argument("--universe", default=str(Path("output") / "etf_universe_snapshot.csv"))
    apply_top100.add_argument("--historical-review", default=str(Path("output") / "ml_sim_historical_review_filled.csv"))
    apply_top100.add_argument("--prices", default=str(Path("data") / "etf_daily.csv"))

    stability = sub.add_parser("ml-stability", help="run offline ML stability diagnostics")
    stability.add_argument("--samples", required=True, help="entry_candidate_samples_labeled CSV/parquet")
    stability.add_argument("--out", required=True)

    recommendations = sub.add_parser("make-recommendations", help="build offline entry manual review recommendations")
    recommendations.add_argument("--artifacts", required=True, help="historical_ml artifacts directory")
    recommendations.add_argument("--out", required=True)
    return p


def _config_from_args(args) -> HistoricalMLConfig:
    return HistoricalMLConfig(
        replay_start=__import__("datetime").date.fromisoformat(args.start) if hasattr(args, "start") else HistoricalMLConfig().replay_start,
        replay_end=__import__("datetime").date.fromisoformat(args.end) if hasattr(args, "end") else HistoricalMLConfig().replay_end,
        output_format=getattr(args, "format", "csv"),
        market_index_code=getattr(args, "market_code", None),
    )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "replay":
        total_start = perf_counter()
        timing = empty_timing()
        config = _config_from_args(args)
        load_start = perf_counter()
        prices = read_price_data(args.prices)
        timing["load_data_seconds"] = perf_counter() - load_start
        engine = HistoricalReplayEngine(prices, config=config)
        outputs = engine.run(config.replay_start, config.replay_end, out_dir=out_dir)
        timing.update({k: v for k, v in engine.last_timing.items() if k in timing or k.endswith("_seconds")})
        timing["total_seconds"] = perf_counter() - total_start
        write_performance_outputs(out_dir, timing, command="replay", row_counts={k: len(v) for k, v in outputs.items()})
        return 0

    if args.command == "label":
        config = HistoricalMLConfig(output_format=args.format, market_index_code=args.market_code)
        prices = read_price_data(args.prices)
        samples = read_table(args.samples)
        labeled = FutureLabeler(prices, config=config).attach_labels(samples)
        write_table(labeled, out_dir, "entry_candidate_samples_labeled", args.format)
        return 0

    if args.command == "label-universe":
        total_start = perf_counter()
        timing = empty_timing()
        config = HistoricalMLConfig(output_format=args.format, market_index_code=args.market_code)
        load_start = perf_counter()
        prices = read_price_data(args.prices)
        samples = read_table(args.samples)
        timing["load_data_seconds"] = perf_counter() - load_start
        horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
        label_start = perf_counter()
        labeler = MLEntryUniverseLabeler(prices, config=config, label_horizons=horizons)
        labeled = labeler.attach_labels(samples)
        labeler.write_outputs(labeled, out_dir)
        timing["label_build_seconds"] = perf_counter() - label_start
        timing["total_seconds"] = perf_counter() - total_start
        write_performance_outputs(out_dir, timing, command="label-universe", row_counts={"ml_entry_labeled_samples": len(labeled)})
        return 0

    if args.command == "report":
        config = HistoricalMLConfig(output_format=args.format)
        labeled = read_table(args.labeled_samples)
        review = build_manual_review_queue(labeled, config=config)
        write_table(review, out_dir, "manual_review_queue", args.format)
        audit_inputs = {
            "daily_etf_samples": read_table(args.daily_etf_samples) if args.daily_etf_samples else None,
            "daily_sector_samples": read_table(args.daily_sector_samples) if args.daily_sector_samples else None,
            "daily_decision_snapshot": read_table(args.daily_decision_snapshot) if args.daily_decision_snapshot else None,
            "entry_candidate_samples_unlabeled": read_table(args.unlabeled_samples) if args.unlabeled_samples else None,
        }
        audit_inputs = {k: v for k, v in audit_inputs.items() if v is not None}
        if audit_inputs:
            generate_replay_audit_report(audit_inputs, labeled, out_dir / "replay_audit_report.md", config=config)
        generate_entry_threshold_report(labeled, out_dir / "entry_threshold_report.md", config=config)
        generate_entry_calibration_outputs(labeled, out_dir, config=config)
        return 0

    if args.command == "run-all":
        total_start = perf_counter()
        timing = empty_timing()
        config = _config_from_args(args)
        load_start = perf_counter()
        prices = read_price_data(args.prices)
        timing["load_data_seconds"] = perf_counter() - load_start
        engine = HistoricalReplayEngine(prices, config=config)
        outputs = engine.run(config.replay_start, config.replay_end, out_dir=out_dir)
        timing.update({k: v for k, v in engine.last_timing.items() if k in timing or k.endswith("_seconds")})
        label_start = perf_counter()
        label_prices = _trim_prices_by_trading_window(
            prices,
            config.replay_start,
            config.replay_end,
            after=max(config.label_horizons) + 1,
        )
        future_labeler = FutureLabeler(label_prices, config=config)
        labeled = future_labeler.attach_labels(outputs["entry_candidate_samples"])
        write_table(labeled, out_dir, "entry_candidate_samples_labeled", config.output_format)
        label_cache = (future_labeler.label_frame, future_labeler.sector_return_frame, future_labeler.label_dates)
        universe_labeler = MLEntryUniverseLabeler(prices, config=config, precomputed_label_cache=label_cache)
        universe_labeled = universe_labeler.attach_labels(outputs["daily_ml_universe_samples"])
        universe_labeler.write_outputs(universe_labeled, out_dir)
        timing["label_build_seconds"] = perf_counter() - label_start
        review = build_manual_review_queue(labeled, config=config)
        write_table(review, out_dir, "manual_review_queue", config.output_format)
        audit_outputs = {
            "daily_etf_samples": outputs["daily_etf_samples"],
            "daily_ml_universe_samples": outputs["daily_ml_universe_samples"],
            "daily_sector_samples": outputs["daily_sector_samples"],
            "daily_decision_snapshot": outputs["daily_decision_snapshot"],
            "entry_candidate_samples_unlabeled": outputs["entry_candidate_samples"],
        }
        generate_replay_audit_report(audit_outputs, labeled, out_dir / "replay_audit_report.md", config=config)
        generate_entry_threshold_report(labeled, out_dir / "entry_threshold_report.md", config=config)
        generate_entry_calibration_outputs(labeled, out_dir, config=config)
        if args.with_ml:
            model_start = perf_counter()
            run_baseline(labeled, out_dir, target="both", output_format=config.output_format)
            timing["model_train_seconds"] = perf_counter() - model_start
        timing["total_seconds"] = perf_counter() - total_start
        write_performance_outputs(
            out_dir,
            timing,
            command="run-all",
            row_counts={**{k: len(v) for k, v in outputs.items()}, "entry_candidate_samples_labeled": len(labeled), "ml_entry_labeled_samples": len(universe_labeled)},
        )
        return 0

    if args.command == "train-baseline":
        run_baseline_from_file(args.samples, out_dir, target=args.target, output_format=args.format)
        return 0

    if args.command == "train-entry-model":
        train_entry_quality_model_from_file(
            args.samples,
            out_dir,
            output_format=args.format,
            top_n=args.top_n,
            min_train_dates=args.min_train_dates,
        )
        return 0

    if args.command == "score-entry-universe":
        total_start = perf_counter()
        timing = empty_timing()
        load_start = perf_counter()
        labeled_samples = read_table(args.labeled_samples)
        scoring_samples = _read_scoring_samples(args.scoring_samples, args.score_date)
        timing["load_data_seconds"] = perf_counter() - load_start
        score_start = perf_counter()
        scores = score_entry_quality_universe(
            labeled_samples,
            scoring_samples,
            out_dir=out_dir,
            output_format=args.format,
            score_date=args.score_date,
            min_train_dates=args.min_train_dates,
        )
        timing["score_inference_seconds"] = perf_counter() - score_start
        timing["total_seconds"] = perf_counter() - total_start
        write_performance_outputs(out_dir, timing, command="score-entry-universe", row_counts={"ml_entry_scores": len(scores)})
        return 0

    if args.command == "coverage-report":
        generate_ml_universe_coverage_report(
            prices_path=args.prices,
            pre_selection_path=args.pre_selection,
            ml_scores_path=args.scores,
            daily_universe_path=args.daily_universe,
            entry_signal_path=args.entry_signal,
            order_intent_path=args.order_intent,
            out_dir=out_dir,
            trade_date=args.trade_date,
        )
        return 0

    if args.command == "compare-policies":
        compare_ml_policies_from_files(
            labeled_samples_path=args.samples,
            ml_scores_path=args.scores,
            out_dir=out_dir,
            output_format=args.format,
        )
        return 0

    if args.command == "ml-sim-review":
        result = build_ml_sim_weekly_review_from_files(
            review_queue_path=args.review_queue,
            prices_path=args.prices,
            out_dir=out_dir,
            comparison_path=args.comparison,
            summary_path=args.summary,
            daily_decision_snapshot_path=args.daily_decision_snapshot,
        )
        print(result.output_paths["weekly_report_md"])
        return 0

    if args.command == "ml-sim-historical-review":
        result = build_ml_sim_historical_review_from_files(
            labeled_samples_path=args.labeled_samples,
            scoring_samples_path=args.scoring_samples,
            prices_path=args.prices,
            out_dir=out_dir,
            start=args.start,
            end=args.end,
            recent_days=args.recent_days,
            min_train_dates=args.min_train_dates,
        )
        print(result.output_paths["report_md"])
        return 0

    if args.command == "ml-recovered-thresholds":
        result = build_ml_recovered_threshold_recommendation_from_file(
            historical_review_path=args.historical_review,
            out_dir=out_dir,
        )
        print(result.output_paths["report_md"])
        return 0

    if args.command == "ml-core-recovered-review":
        result = build_ml_core_recovered_review_from_file(
            historical_review_path=args.historical_review,
            out_dir=out_dir,
        )
        print(result.output_paths["recommendation_md"])
        return 0

    if args.command == "sector-mapping-audit":
        result = build_sector_mapping_audit_from_files(
            historical_review_path=args.historical_review,
            out_dir=out_dir,
            sector_map_path=args.sector_map,
            universe_path=args.universe,
            price_path=args.prices,
        )
        print(result.output_paths["after_audit_md"])
        return 0

    if args.command == "sector-mapping-suggestion":
        result = build_sector_mapping_suggestions_from_files(
            sector_map_path=args.sector_map,
            universe_path=args.universe,
            out_dir=out_dir,
            price_path=args.prices,
            strong_audit_path=args.strong_audit,
            historical_review_path=args.historical_review,
            current_position_path=args.current_position,
            entry_signal_path=args.entry_signal,
        )
        print(result.output_paths["coverage_md"])
        return 0

    if args.command == "sector-mapping-top100-review":
        result = build_sector_mapping_top100_review_from_files(
            priority_review_path=args.priority_review,
            sector_map_path=args.sector_map,
            out_dir=out_dir,
            universe_path=args.universe,
            price_path=args.prices,
            strong_audit_path=args.strong_audit,
            historical_review_path=args.historical_review,
            entry_signal_path=args.entry_signal,
            top_n=args.top_n,
        )
        print(result.output_paths["summary_md"])
        return 0

    if args.command == "sector-mapping-apply-top100":
        result = apply_sector_mapping_top100_from_files(
            sector_map_path=args.sector_map,
            review_path=args.review,
            patch_draft_path=args.patch_draft,
            out_dir=out_dir,
            universe_path=args.universe,
            historical_review_path=args.historical_review,
            price_path=args.prices,
        )
        print(result.output_paths["summary_md"])
        return 0

    if args.command == "ml-stability":
        run_ml_stability_from_file(args.samples, out_dir)
        return 0

    if args.command == "make-recommendations":
        make_recommendations_from_artifacts(args.artifacts, out_dir)
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""historical_ml: historical replay sample factory for ETF entry calibration."""

from .config import HistoricalMLConfig
from .entry_adapter import RealEntryAdapter
from .audit import generate_replay_audit_report, validate_replay_outputs
from .calibration import build_entry_calibration, generate_entry_calibration_outputs
from .daily_universe import build_daily_ml_universe_samples, build_daily_ml_universe_summary
from .replay_engine import HistoricalReplayEngine
from .labeler import FutureLabeler, MLEntryUniverseLabeler
from .ml_entry_model import train_entry_quality_model, train_entry_quality_model_from_file
from .ml_core_recovered_review import build_ml_core_recovered_review, build_ml_core_recovered_review_from_file
from .ml_recovered_thresholds import build_ml_recovered_threshold_recommendation, build_ml_recovered_threshold_recommendation_from_file
from .ml_sim_historical_review import build_ml_sim_historical_review, build_ml_sim_historical_review_from_files
from .ml_sim_review import build_ml_sim_weekly_review, build_ml_sim_weekly_review_from_files
from .policy_comparison import compare_ml_policies, compare_ml_policies_from_files
from .review_queue import build_manual_review_queue
from .reports import generate_entry_threshold_report
from .sector_mapping_audit import build_sector_mapping_audit, build_sector_mapping_audit_from_files
from .sector_mapping_suggestion import (
    apply_sector_mapping_top100_from_files,
    build_sector_mapping_suggestions,
    build_sector_mapping_suggestions_from_files,
    build_sector_mapping_top100_review,
    build_sector_mapping_top100_review_from_files,
)

__all__ = [
    "HistoricalMLConfig",
    "RealEntryAdapter",
    "generate_replay_audit_report",
    "validate_replay_outputs",
    "build_entry_calibration",
    "generate_entry_calibration_outputs",
    "build_daily_ml_universe_samples",
    "build_daily_ml_universe_summary",
    "HistoricalReplayEngine",
    "FutureLabeler",
    "MLEntryUniverseLabeler",
    "train_entry_quality_model",
    "train_entry_quality_model_from_file",
    "build_ml_core_recovered_review",
    "build_ml_core_recovered_review_from_file",
    "build_ml_recovered_threshold_recommendation",
    "build_ml_recovered_threshold_recommendation_from_file",
    "build_ml_sim_historical_review",
    "build_ml_sim_historical_review_from_files",
    "build_ml_sim_weekly_review",
    "build_ml_sim_weekly_review_from_files",
    "compare_ml_policies",
    "compare_ml_policies_from_files",
    "build_manual_review_queue",
    "generate_entry_threshold_report",
    "build_sector_mapping_audit",
    "build_sector_mapping_audit_from_files",
    "build_sector_mapping_suggestions",
    "build_sector_mapping_suggestions_from_files",
    "build_sector_mapping_top100_review",
    "build_sector_mapping_top100_review_from_files",
    "apply_sector_mapping_top100_from_files",
]

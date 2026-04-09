"""
Scoring & Training API routes — protected by Clerk auth, scoped by tenant.
Uses raw SQL against Turso/libSQL for all database operations.
"""

import io
import json
import uuid
import logging
from typing import Optional, List, Dict

from fastapi import APIRouter, UploadFile, File, Query, Depends, BackgroundTasks
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from app.database import get_db
from app.core.auth import get_current_user
from app.core.config import get_settings
from app.core.responses import success_response, error_response
from app.services import model_storage
from app.services.dataset_relationships import (
    DatasetAsset,
    analyze_dataset_collection,
    prepare_combined_dataset,
)
from app.services.explanation_translator import translate_scoring_results
from adaptive_scorer import UniversalAdaptiveScorer, DataAnalyzer, EngagementScorer, ActionRecommender

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["Scoring"])

# ── In-memory model cache ─────────────────────────────────────
# {tenant_id: {model_name: UniversalAdaptiveScorer}}
trained_models = {}


def _row_signature(data: dict) -> str:
    """Build a deterministic fingerprint for a scored row."""
    normalized = {}
    for key, value in sorted(data.items()):
        if isinstance(value, float):
            normalized[key] = round(value, 6)
        else:
            normalized[key] = value
    return json.dumps(normalized, sort_keys=True, default=str)


def _get_model(tenant_id: str, model_name: str) -> Optional[UniversalAdaptiveScorer]:
    """Get model from cache."""
    return trained_models.get(tenant_id, {}).get(model_name)


def _set_model(tenant_id: str, model_name: str, model: UniversalAdaptiveScorer):
    """Put model in cache."""
    if tenant_id not in trained_models:
        trained_models[tenant_id] = {}
    trained_models[tenant_id][model_name] = model


def init_models_cache(all_models):
    """Initialize cache on startup from loaded models."""
    global trained_models
    trained_models = all_models


# ── Helper: Smart Merge ──────────────────────────────────────

def smart_merge_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Intelligently merge multiple DataFrames."""
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0]

    cols_set = [set(df.columns) for df in dfs]
    if all(s == cols_set[0] for s in cols_set):
        return pd.concat(dfs, axis=0, ignore_index=True)

    common_cols = set.intersection(*cols_set)
    candidate_keys = []
    for col in common_cols:
        is_id_name = any(kw in col.lower() for kw in ['id', 'email', 'key', 'contact', 'prospect'])
        is_unique = all(df[col].nunique() / len(df) > 0.8 for df in dfs if len(df) > 0)
        if is_id_name or is_unique:
            candidate_keys.append(col)

    if candidate_keys:
        key = sorted(candidate_keys, key=lambda x: ('email' in x.lower() or 'id' in x.lower()), reverse=True)[0]
        logger.info("MERGE: Joining on key: %s", key)
        merged = dfs[0]
        for i in range(1, len(dfs)):
            merged = pd.merge(merged, dfs[i], on=key, how='left', suffixes=('', f'_extra_{i}'))
        return merged

    logger.info("MERGE: No common ID key found. Concatenating.")
    return pd.concat(dfs, axis=0, ignore_index=True)


# ── Input Validation ─────────────────────────────────────────

MAX_CSV_SIZE_MB = settings.MAX_CSV_SIZE_MB
MAX_CSV_SIZE = MAX_CSV_SIZE_MB * 1024 * 1024
MAX_COLUMNS = 500


async def _validate_and_read_files(
    file: Optional[UploadFile],
    files: Optional[List[UploadFile]],
) -> List[pd.DataFrame]:
    """Validate and read uploaded CSV files with security checks."""
    all_files = []
    if file:
        all_files.append(file)
    if files:
        all_files.extend(files)

    if not all_files:
        raise ValueError("No files uploaded. Use field 'file' or 'files'.")

    dfs = []
    for f in all_files:
        # Validate file extension
        filename = f.filename or ""
        if not filename.lower().endswith(".csv"):
            raise ValueError(f"Only .csv files accepted. Got: '{filename}'")

        contents = await f.read()

        # Validate file size
        if len(contents) > MAX_CSV_SIZE:
            raise ValueError(
                f"File '{filename}' exceeds {MAX_CSV_SIZE_MB}MB limit ({len(contents) / 1024 / 1024:.1f}MB)"
            )

        df = pd.read_csv(io.BytesIO(contents))

        # Validate column count
        if len(df.columns) > MAX_COLUMNS:
            raise ValueError(f"File '{filename}' has {len(df.columns)} columns (max {MAX_COLUMNS})")

        dfs.append(df)

    return dfs


def _uploaded_dataset_names(
    file: Optional[UploadFile],
    files: Optional[List[UploadFile]],
) -> List[str]:
    uploaded = []
    if file:
        uploaded.append(file)
    if files:
        uploaded.extend(files)

    names = []
    for index, uploaded_file in enumerate(uploaded, start=1):
        names.append(uploaded_file.filename or f"dataset_{index}.csv")
    return names


def _prepare_assets(dataset_names: List[str], dfs: List[pd.DataFrame]) -> List[DatasetAsset]:
    return [
        DatasetAsset(name=name, df=df)
        for name, df in zip(dataset_names, dfs)
    ]


# ── Background task: persist scored leads ────────────────────

def _persist_scores(tenant_id: str, model_name: str, results: list):
    """Save scored leads to Turso DB in background (doesn't slow down response)."""
    try:
        conn = get_db()

        # Find training run for this model
        result = conn.execute(
            "SELECT id FROM training_runs WHERE tenant_id = ? AND model_name = ? ORDER BY created_at DESC LIMIT 1",
            [tenant_id, model_name],
        )
        training_run_id = result.rows[0][0] if result.rows else None

        for r in results:
            lead_id = str(uuid.uuid4())
            score = r.get("score", 0.0)
            lead_data = json.dumps(r.get("data", {}), default=str)
            lead_signature = _row_signature(r.get("data", {}))
            ranking_version = r.get("ranking_version")

            conn.execute(
                """INSERT INTO scored_leads (
                       id, tenant_id, training_run_id, lead_data, lead_signature,
                       model_name, ranking_version, final_score, scored_at
                   )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                [lead_id, tenant_id, training_run_id, lead_data, lead_signature, model_name, ranking_version, score],
            )

        logger.info("Persisted %d scored leads for tenant=%s", len(results), tenant_id)
    except Exception as e:
        logger.error("Failed to persist scores: %s", str(e))


@router.post("/merge-plan")
async def merge_plan(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    user: dict = Depends(get_current_user),
):
    """Profile uploaded datasets and recommend safe relationship-aware merge steps."""
    try:
        dataset_names = _uploaded_dataset_names(file, files)
        dfs = await _validate_and_read_files(file, files)
        assets = _prepare_assets(dataset_names, dfs)
        analysis = analyze_dataset_collection(assets)
        _, plan = prepare_combined_dataset(assets)

        return success_response(data={
            "status": "success",
            "analysis": analysis,
            "merge_plan": plan,
        })
    except ValueError as e:
        return error_response("VALIDATION_ERROR", str(e), 400)
    except Exception as e:
        logger.exception("Merge planning failed")
        return error_response("MERGE_PLAN_FAILED", f"Merge planning failed: {str(e)}", 500)


def _get_model_version_history(tenant_id: str, model_name: str, limit: int = 2):
    """Return latest training run rows for a model."""
    conn = get_db()
    result = conn.execute(
        """SELECT id, artifact_path, created_at
           FROM training_runs
           WHERE tenant_id = ? AND model_name = ?
           ORDER BY created_at DESC
           LIMIT ?""",
        [tenant_id, model_name, limit],
    )
    return result.rows if result.rows else []


def _load_feedback_training_frame(tenant_id: str, model_name: str, target_column: str) -> pd.DataFrame:
    """Reconstruct a supervised training frame from persisted feedback events."""
    conn = get_db()
    rows = conn.execute(
        """SELECT lead_data, actual_outcome
           FROM feedback_events
           WHERE tenant_id = ? AND model_name = ?
           ORDER BY feedback_at DESC""",
        [tenant_id, model_name],
    ).rows

    records = []
    for lead_data_json, actual_outcome in rows:
        try:
            lead_data = json.loads(lead_data_json) if lead_data_json else {}
        except json.JSONDecodeError:
            continue
        lead_data[target_column] = int(actual_outcome)
        records.append(lead_data)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def _load_segment_feedback_training_frame(
    tenant_id: str,
    model_name: str,
    target_column: str,
    segment_dimension: str,
    segment_value: str,
) -> pd.DataFrame:
    """Reconstruct a supervised training frame for one drifting segment."""
    feedback_df = _load_feedback_training_frame(tenant_id, model_name, target_column)
    if feedback_df.empty or segment_dimension not in feedback_df.columns:
        return pd.DataFrame()

    filtered = feedback_df[feedback_df[segment_dimension].astype(str) == str(segment_value)].copy()
    if filtered.empty:
        return pd.DataFrame()
    return filtered


def _auto_retrain_policy(learning_signal: Dict) -> Dict:
    """Decide whether feedback evidence is strong enough to auto-retrain."""
    matched_rows = int(learning_signal.get("matched_rows", 0))
    feedback_accuracy = float(learning_signal.get("feedback_accuracy", 0.0))
    feedback_recall = float(learning_signal.get("feedback_recall", 0.0))
    recommendation = learning_signal.get("recommendation")

    should_retrain = (
        matched_rows >= 25
        and recommendation == "retrain_with_feedback"
        and (feedback_accuracy < 0.82 or feedback_recall < 0.72)
    )

    reasons = []
    if matched_rows >= 25:
        reasons.append("enough_feedback_volume")
    if feedback_accuracy < 0.82:
        reasons.append("accuracy_below_policy")
    if feedback_recall < 0.72:
        reasons.append("recall_below_policy")

    return {
        "should_auto_retrain": should_retrain,
        "policy_name": "feedback_guardrail_v1",
        "reasons": reasons,
        "thresholds": {
            "min_matched_rows": 25,
            "min_accuracy": 0.82,
            "min_recall": 0.72,
        },
    }


def _execute_feedback_retrain(
    tenant_id: str,
    model_name: str,
    feedback_weight: int,
    scorer: UniversalAdaptiveScorer,
):
    """Retrain a model from persisted feedback events and persist the new version."""
    target_column = scorer.analyzer.target_col
    feedback_df = _load_feedback_training_frame(tenant_id, model_name, target_column)

    if feedback_df.empty:
        raise ValueError(f"No feedback events found for model '{model_name}'. Upload outcomes first.")
    if len(feedback_df) < 10:
        raise ValueError(f"Need at least 10 matched feedback rows to retrain. Found {len(feedback_df)}.")

    weighted_frames = [feedback_df.copy() for _ in range(feedback_weight)]
    retrain_df = pd.concat(weighted_frames, ignore_index=True)

    new_scorer = UniversalAdaptiveScorer()
    result = new_scorer.train(
        retrain_df,
        target_col=target_column,
        client_id=model_name,
    )

    artifact_path = model_storage.save_model(new_scorer, tenant_id, model_name)
    _set_model(tenant_id, model_name, new_scorer)

    run_id = str(uuid.uuid4())
    conn = get_db()
    metrics_payload = {
        **result["metrics"],
        "training_source": "feedback_events",
        "feedback_rows": int(len(feedback_df)),
        "feedback_weight": int(feedback_weight),
    }
    if new_scorer.scorer:
        new_scorer.scorer.metadata.update(metrics_payload)
    conn.execute(
        """INSERT INTO training_runs (id, tenant_id, model_name, artifact_path, metrics, row_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
        [run_id, tenant_id, model_name, artifact_path, json.dumps(metrics_payload, default=str), len(retrain_df)],
    )

    logger.info(
        "Feedback retrain complete: tenant=%s model=%s rows=%d weighted_rows=%d",
        tenant_id,
        model_name,
        len(feedback_df),
        len(retrain_df),
    )

    return {
        "status": "success",
        "model_name": model_name,
        "message": f"Retrained from {len(feedback_df)} feedback rows with weight {feedback_weight}",
        "analysis": result["analysis"],
        "metrics": metrics_payload,
    }


def _execute_segment_feedback_retrain(
    tenant_id: str,
    model_name: str,
    feedback_weight: int,
    scorer: UniversalAdaptiveScorer,
    segment_dimension: str,
    segment_value: str,
):
    """Retrain a model from feedback rows belonging to one segment hotspot."""
    target_column = scorer.analyzer.target_col
    feedback_df = _load_segment_feedback_training_frame(
        tenant_id,
        model_name,
        target_column,
        segment_dimension,
        segment_value,
    )

    if feedback_df.empty:
        raise ValueError(
            f"No feedback events found for segment '{segment_dimension}={segment_value}' on model '{model_name}'."
        )
    if len(feedback_df) < 8:
        raise ValueError(
            f"Need at least 8 matched feedback rows for segment retrain. Found {len(feedback_df)}."
        )

    weighted_frames = [feedback_df.copy() for _ in range(feedback_weight)]
    retrain_df = pd.concat(weighted_frames, ignore_index=True)

    segment_model_name = f"{model_name}__{segment_dimension}_{str(segment_value).replace(' ', '_')}"
    new_scorer = UniversalAdaptiveScorer()
    result = new_scorer.train(
        retrain_df,
        target_col=target_column,
        client_id=segment_model_name,
    )

    artifact_path = model_storage.save_model(new_scorer, tenant_id, segment_model_name)
    _set_model(tenant_id, segment_model_name, new_scorer)

    run_id = str(uuid.uuid4())
    conn = get_db()
    metrics_payload = {
        **result["metrics"],
        "training_source": "segment_feedback_events",
        "feedback_rows": int(len(feedback_df)),
        "feedback_weight": int(feedback_weight),
        "segment_dimension": segment_dimension,
        "segment_value": str(segment_value),
        "base_model_name": model_name,
    }
    if new_scorer.scorer:
        new_scorer.scorer.metadata.update(metrics_payload)
    conn.execute(
        """INSERT INTO training_runs (id, tenant_id, model_name, artifact_path, metrics, row_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
        [
            run_id,
            tenant_id,
            segment_model_name,
            artifact_path,
            json.dumps(metrics_payload, default=str),
            len(retrain_df),
        ],
    )

    logger.info(
        "Segment feedback retrain complete: tenant=%s base_model=%s segment=%s=%s rows=%d weighted_rows=%d",
        tenant_id,
        model_name,
        segment_dimension,
        segment_value,
        len(feedback_df),
        len(retrain_df),
    )

    return {
        "status": "success",
        "model_name": segment_model_name,
        "message": f"Retrained segment model for {segment_dimension}={segment_value} from {len(feedback_df)} feedback rows",
        "analysis": result["analysis"],
        "metrics": metrics_payload,
    }


def _get_segment_models_for_base(tenant_id: str, base_model_name: str):
    """Return cached segment-specialized models derived from a base model."""
    segment_models = []
    tenant_models = trained_models.get(tenant_id, {})
    for candidate_name, candidate_model in tenant_models.items():
        if candidate_name == base_model_name:
            continue
        scorer = getattr(candidate_model, "scorer", None)
        metadata = getattr(scorer, "metadata", {}) if scorer else {}
        if metadata.get("base_model_name") != base_model_name:
            continue
        if not metadata.get("segment_dimension") or metadata.get("segment_value") is None:
            continue
        segment_models.append({
            "model_name": candidate_name,
            "model": candidate_model,
            "segment_dimension": metadata.get("segment_dimension"),
            "segment_value": str(metadata.get("segment_value")),
            "feedback_rows": int(metadata.get("feedback_rows", 0) or 0),
            "accuracy": float(metadata.get("accuracy", 0.0) or 0.0),
            "roc_auc": float(metadata.get("roc_auc", 0.0) or 0.0),
        })
    return segment_models


def _route_priority(candidate: Dict) -> float:
    """Deterministic priority score for choosing between matching segment models."""
    return (
        candidate.get("feedback_rows", 0) * 1.0
        + candidate.get("roc_auc", 0.0) * 100
        + candidate.get("accuracy", 0.0) * 10
    )


def _route_and_score_rows(
    tenant_id: str,
    base_model_name: str,
    base_scorer: UniversalAdaptiveScorer,
    df: pd.DataFrame,
):
    """Score rows with segment-specialized models when they match, otherwise use base model."""
    segment_models = _get_segment_models_for_base(tenant_id, base_model_name)
    if not segment_models:
        results = base_scorer.score(df)
        for result in results:
            result["routing"] = {
                "used_model": base_model_name,
                "route_type": "base",
                "policy": "lucida_route_policy_v1",
                "reason": "no_segment_models_available",
                "candidates_considered": [],
            }
        return results

    routed_results = []
    for idx in range(len(df)):
        row_df = df.iloc[[idx]].copy()
        row_series = row_df.iloc[0]
        selected = {
            "used_model": base_model_name,
            "route_type": "base",
            "matched_segment": None,
            "policy": "lucida_route_policy_v1",
            "reason": "no_segment_match",
            "candidates_considered": [],
        }
        scorer_to_use = base_scorer
        matching_candidates = []

        for candidate in segment_models:
            dimension = candidate["segment_dimension"]
            if dimension not in row_df.columns:
                continue
            row_value = row_series.get(dimension)
            if pd.isna(row_value):
                continue
            if str(row_value) == candidate["segment_value"]:
                matching_candidates.append(candidate)

        if matching_candidates:
            ranked_candidates = sorted(
                matching_candidates,
                key=_route_priority,
                reverse=True,
            )
            chosen = ranked_candidates[0]
            scorer_to_use = chosen["model"]
            selected = {
                "used_model": chosen["model_name"],
                "route_type": "segment",
                "matched_segment": {
                    "dimension": chosen["segment_dimension"],
                    "value": chosen["segment_value"],
                },
                "policy": "lucida_route_policy_v1",
                "reason": "highest_priority_matching_segment",
                "candidates_considered": [
                    {
                        "model_name": candidate["model_name"],
                        "segment_dimension": candidate["segment_dimension"],
                        "segment_value": candidate["segment_value"],
                        "feedback_rows": candidate["feedback_rows"],
                        "accuracy": round(candidate["accuracy"], 4),
                        "roc_auc": round(candidate["roc_auc"], 4),
                        "priority_score": round(_route_priority(candidate), 4),
                    }
                    for candidate in ranked_candidates
                ],
            }

        result = scorer_to_use.score(row_df)[0]
        result["routing"] = selected
        routed_results.append(result)

    routed_results.sort(key=lambda item: item["score"], reverse=True)
    return routed_results


def _compare_against_previous_version(
    tenant_id: str,
    model_name: str,
    df: pd.DataFrame,
    current_results: list,
):
    """Compare current ranking to the previous saved model version for the same model name."""
    rows = _get_model_version_history(tenant_id, model_name, limit=2)
    if len(rows) < 2:
        return None

    previous_artifact_path = rows[1][1]
    previous_created_at = rows[1][2]

    try:
        previous_model = model_storage.load_model_from_path(previous_artifact_path)
        previous_results = previous_model.score(df.copy())
    except Exception as exc:
        logger.warning("Unable to compare previous version for tenant=%s model=%s: %s", tenant_id, model_name, exc)
        return None

    previous_index = {}
    for rank, result in enumerate(previous_results, start=1):
        previous_index[_row_signature(result.get("data", {}))] = {
            "rank": rank,
            "score": result.get("score", 0.0),
        }

    for rank, result in enumerate(current_results, start=1):
        signature = _row_signature(result.get("data", {}))
        previous = previous_index.get(signature)
        if not previous:
            result["rank_movement"] = {
                "status": "new",
                "current_rank": rank,
                "previous_rank": None,
                "rank_delta": None,
                "score_delta": None,
            }
            continue

        rank_delta = previous["rank"] - rank
        score_delta = round(float(result.get("score", 0.0) - previous.get("score", 0.0)), 2)

        status = "unchanged"
        if rank_delta > 0:
            status = "up"
        elif rank_delta < 0:
            status = "down"

        result["rank_movement"] = {
            "status": status,
            "current_rank": rank,
            "previous_rank": previous["rank"],
            "rank_delta": rank_delta,
            "score_delta": score_delta,
            "compared_to": previous_created_at,
        }

    return {
        "baseline_created_at": previous_created_at,
        "comparison_type": "previous_model_version",
    }


# ── Train ────────────────────────────────────────────────────

@router.post("/train")
async def train_model(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    model_name: str = Query("default", description="Name for this model"),
    target_column: Optional[str] = Query(None),
    mode: str = Query("supervised", description="Training mode: 'supervised' (requires binary target) or 'unsupervised' (ranks rows without labels)"),
    user: dict = Depends(get_current_user),
):
    """Upload CSVs → auto-merge → auto-train. Protected + tenant-scoped.
    
    Modes:
    - supervised (default): Requires a binary target column. Uses adaptive scorer for classification.
    - unsupervised: No target needed. Ranks rows using multi-criteria signals without labels.
    """
    tenant_id = user["tenant_id"]
    try:
        dataset_names = _uploaded_dataset_names(file, files)
        dfs = await _validate_and_read_files(file, files)
        assets = _prepare_assets(dataset_names, dfs)

        logger.info("Training: tenant=%s model=%s mode=%s files=%d", tenant_id, model_name, mode, len(dfs))

        df, merge_plan = prepare_combined_dataset(assets)
        logger.info("Prepared data shape: %s using strategy=%s", df.shape, merge_plan.get("strategy"))

        if len(assets) > 1 and not merge_plan.get("executed_steps") and merge_plan.get("warnings"):
            return error_response(
                "NO_SAFE_MERGE_PLAN",
                "No safe dataset relationship was found. Review merge-plan results or upload datasets with stronger keys.",
                400,
            )

        if df.shape[0] < 10:
            return error_response("TOO_FEW_ROWS", f"Need at least 10 rows, got {df.shape[0]}", 400)
        if df.shape[1] < 2:
            return error_response("TOO_FEW_COLUMNS", f"Need at least 2 columns, got {df.shape[1]}", 400)

        if mode == "unsupervised":
            # ── UNSUPERVISED MODE: Rank rows without needing binary target ──
            scorer = UniversalAdaptiveScorer()
            # Train on dummy binary target to initialize analyzer
            # Use row index parity (even/odd) as synthetic target for initialization only
            df_with_synthetic = df.copy()
            df_with_synthetic["__synthetic_target__"] = (np.arange(len(df)) % 2).astype(int)
            
            train_result = scorer.train(df_with_synthetic, target_col="__synthetic_target__", client_id=model_name)
            
            # Mark this model as unsupervised so scoring knows to rank differently
            if scorer.scorer and hasattr(scorer.scorer, 'metadata'):
                scorer.scorer.metadata['training_mode'] = 'unsupervised'
                scorer.scorer.metadata['original_columns'] = list(df.columns)
            
            train_result["analysis"]["training_mode"] = "unsupervised"
            train_result["analysis"]["message"] = "Trained in UNSUPERVISED mode: ranks rows by multi-criteria signals (no binary target required)"

        else:
            # ── SUPERVISED MODE: Standard classification with binary target ──
            scorer = UniversalAdaptiveScorer()
            train_result = scorer.train(df, target_col=target_column, client_id=model_name)

            if not target_column and train_result["analysis"]["target_diagnostics"].get("recommendation") == "manual_review_recommended":
                return error_response(
                    "AMBIGUOUS_TARGET",
                    "Automatic target detection is not reliable for this CRM export. Please provide target_column explicitly or use mode=unsupervised.",
                    400,
                )
            
            train_result["analysis"]["training_mode"] = "supervised"

        # Save model to disk
        artifact_path = model_storage.save_model(scorer, tenant_id, model_name)

        # Cache in memory
        _set_model(tenant_id, model_name, scorer)

        # Save training run to Turso
        run_id = str(uuid.uuid4())
        conn = get_db()
        conn.execute(
            """INSERT INTO training_runs (id, tenant_id, model_name, artifact_path, metrics, row_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            [run_id, tenant_id, model_name, artifact_path,
             json.dumps(train_result["metrics"], default=str), len(df)],
        )

        logger.info(
            "Training complete: tenant=%s model=%s mode=%s rows=%d",
            tenant_id, model_name, mode, len(df),
        )

        return success_response(data={
            "status": "success",
            "model_name": model_name,
            "training_mode": mode,
            "message": f"Trained on {len(df)} samples ({len(dfs)} files) with {train_result['analysis']['n_features']} features" + 
                      (" - UNSUPERVISED RANKING (no target needed)" if mode == "unsupervised" else ""),
            "analysis": train_result["analysis"],
            "metrics": train_result["metrics"],
            "merge_plan": merge_plan,
        })

    except ValueError as e:
        logger.warning("Training validation error: %s", str(e))
        return error_response("VALIDATION_ERROR", str(e), 400)
    except Exception as e:
        logger.exception("Training failed")
        return error_response("TRAINING_FAILED", f"Training failed: {str(e)}", 500)


# ── Score CSV ────────────────────────────────────────────────

@router.post("/score")
async def score_csv(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    model_name: str = Query("default", description="Model to score with"),
    include_engagement: bool = Query(True, description="Include engagement momentum scoring"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(get_current_user),
):
    """Upload CSVs of leads to score. Protected + tenant-scoped.
    
    Returns dual scores:
    - profile_score: ML-based conversion probability (existing model)
    - engagement_score: Rule-based engagement momentum (auto-detected)
    - recommended_action: Based on 2x2 matrix of both scores
    """
    tenant_id = user["tenant_id"]
    scorer = _get_model(tenant_id, model_name)

    if not scorer:
        return error_response("MODEL_NOT_FOUND", f"No model '{model_name}' found. Train first.", 404)

    try:
        dataset_names = _uploaded_dataset_names(file, files)
        dfs = await _validate_and_read_files(file, files)
        assets = _prepare_assets(dataset_names, dfs)

        df, merge_plan = prepare_combined_dataset(assets)
        results = _route_and_score_rows(tenant_id, model_name, scorer, df)
        rank_tracking = _compare_against_previous_version(tenant_id, model_name, df, results)

        routed_count = sum(1 for row in results if row.get("routing", {}).get("route_type") == "segment")

        # Translate results to sales-friendly language
        enriched_results = translate_scoring_results(results)

        # ── Engagement Scoring (Additive) ─────────────────────────
        engagement_analysis = None
        if include_engagement:
            engagement_scorer = EngagementScorer()
            engagement_analysis = engagement_scorer.analyze(df)
            action_recommender = ActionRecommender()
            
            if engagement_analysis['detected_columns']:
                # Score engagement for each lead
                for i, result in enumerate(enriched_results):
                    row = df.iloc[i] if i < len(df) else None
                    if row is not None:
                        eng_result = engagement_scorer.score_lead(row)
                        profile_score = result.get('score', 0)
                        engagement_score = eng_result.get('engagement_score')
                        
                        # Add engagement data to result
                        result['profile_score'] = profile_score  # Rename for clarity
                        result['engagement_score'] = engagement_score
                        result['engagement_signals'] = eng_result.get('signals', {})
                        result['engagement_band'] = eng_result.get('engagement_band')
                        result['top_engagement_signals'] = eng_result.get('top_signals', [])
                        
                        # Get action recommendation
                        action = action_recommender.recommend(profile_score, engagement_score)
                        result['recommended_action'] = action['action']
                        result['action_emoji'] = action['emoji']
                        result['action_color'] = action['color']
                        result['action_priority'] = action['priority']
                        result['action_description'] = action['description']
                        result['action_next_steps'] = action['next_steps']
                        result['action_confidence'] = action['confidence']
                        result['quadrant'] = action['quadrant']
            else:
                # No engagement columns found - add profile score only with action based on profile
                for result in enriched_results:
                    profile_score = result.get('score', 0)
                    result['profile_score'] = profile_score
                    result['engagement_score'] = None
                    result['engagement_signals'] = {}
                    result['engagement_band'] = None
                    result['top_engagement_signals'] = []
                    
                    action = action_recommender.recommend(profile_score, None)
                    result['recommended_action'] = action['action']
                    result['action_emoji'] = action['emoji']
                    result['action_color'] = action['color']
                    result['action_priority'] = action['priority']
                    result['action_description'] = action['description']
                    result['action_next_steps'] = action['next_steps']
                    result['action_confidence'] = action['confidence']
                    result['quadrant'] = action['quadrant']

        # Persist scores in background (doesn't slow response)
        background_tasks.add_task(_persist_scores, tenant_id, model_name, enriched_results)

        logger.info("Scored %d leads: tenant=%s model=%s", len(enriched_results), tenant_id, model_name)

        response_data = {
            "status": "success",
            "model_name": model_name,
            "n_leads": len(enriched_results),
            "results": enriched_results,
            "rank_tracking": rank_tracking,
            "merge_plan": merge_plan,
            "routing_summary": {
                "base_model": model_name,
                "segment_routed_rows": routed_count,
                "base_routed_rows": len(enriched_results) - routed_count,
            },
        }
        
        # Add engagement summary if included
        if include_engagement and engagement_analysis:
            response_data["engagement_analysis"] = {
                "detected_columns": engagement_analysis['detected_columns'],
                "signals_found": engagement_analysis['signals_found'],
                "signals_missing": engagement_analysis['signals_missing'],
                "coverage_percent": round(engagement_analysis['coverage'], 1),
            }
            
            # Summary stats
            if enriched_results:
                actions_summary = {}
                for r in enriched_results:
                    action = r.get('recommended_action', 'UNKNOWN')
                    actions_summary[action] = actions_summary.get(action, 0) + 1
                response_data["action_summary"] = actions_summary

        return success_response(data=response_data)

    except ValueError as e:
        return error_response("VALIDATION_ERROR", str(e), 400)
    except Exception as e:
        logger.exception("Scoring failed")
        return error_response("SCORING_FAILED", f"Scoring failed: {str(e)}", 500)


# ── Score CSV (legacy alias) ─────────────────────────────────

@router.post("/score-csv")
async def score_csv_legacy(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    model_name: str = Query("default"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(get_current_user),
):
    """Legacy alias for /score. Kept for backward compatibility."""
    return await score_csv(file, files, model_name, background_tasks, user)


# ── Analyze ──────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_csv(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Preview CSV analysis without training. Protected."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        analyzer = DataAnalyzer(df)
        column_types = analyzer.infer_column_types()
        target = analyzer.auto_detect_target()
        importance = analyzer.compute_feature_importance()

        return success_response(data={
            "status": "success",
            "rows": len(df),
            "columns": len(df.columns),
            "column_types": column_types,
            "detected_target": target,
            "target_diagnostics": analyzer.get_target_diagnostics(),
            "feature_importance": dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]),
        })

    except Exception as e:
        return error_response("ANALYSIS_FAILED", f"Analysis failed: {str(e)}", 500)


@router.post("/feedback")
async def ingest_feedback(
    file: UploadFile = File(...),
    model_name: str = Query("default"),
    outcome_column: Optional[str] = Query(None, description="Optional binary outcome column"),
    auto_retrain: bool = Query(False, description="Automatically retrain if policy thresholds are met"),
    feedback_weight: int = Query(2, ge=1, le=10, description="Weight to use if auto retrain runs"),
    user: dict = Depends(get_current_user),
):
    """Upload actual outcome data to measure how past scores performed."""
    tenant_id = user["tenant_id"]
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if df.empty:
            return error_response("EMPTY_FEEDBACK_FILE", "Feedback CSV is empty.", 400)

        analyzer = DataAnalyzer(df, target_col=outcome_column)
        analyzer.infer_column_types()
        detected_target = analyzer.auto_detect_target()
        encoded_outcomes = analyzer._encode_binary(detected_target)

        conn = get_db()
        recent_scores = conn.execute(
            """SELECT lead_signature, lead_data, final_score, scored_at
               FROM scored_leads
               WHERE tenant_id = ? AND model_name = ?
               ORDER BY scored_at DESC
               LIMIT 5000""",
            [tenant_id, model_name],
        ).rows

        if not recent_scores:
            return error_response("NO_SCORES_FOUND", f"No scored leads found for model '{model_name}'. Score leads before uploading feedback.", 404)

        score_index = {}
        for idx, row in enumerate(recent_scores, start=1):
            signature = row[0]
            score_index.setdefault(signature, {
                "lead_data": json.loads(row[1]) if row[1] else {},
                "predicted_score": float(row[2]) if row[2] is not None else 0.0,
                "scored_at": row[3],
                "rank_at_score_time": idx,
            })

        matched_feedback = []
        for idx, row in df.iterrows():
            row_data = row.to_dict()
            outcome_value = int(encoded_outcomes.iloc[idx])
            feature_payload = {}
            for key, value in row_data.items():
                if key == detected_target:
                    continue
                if hasattr(value, "item"):
                    feature_payload[key] = value.item()
                elif pd.isna(value):
                    feature_payload[key] = None
                else:
                    feature_payload[key] = value

            signature = _row_signature(feature_payload)
            prior_score = score_index.get(signature)
            if prior_score:
                matched_feedback.append({
                    "lead_signature": signature,
                    "lead_data": feature_payload,
                    "actual_outcome": outcome_value,
                    "predicted_score": prior_score["predicted_score"],
                    "rank_at_score_time": prior_score["rank_at_score_time"],
                    "scored_at": prior_score["scored_at"],
                })

        if not matched_feedback:
            return error_response(
                "NO_FEEDBACK_MATCHES",
                "No feedback rows matched previously scored leads. Upload the same lead fields used during scoring.",
                400,
            )

        predicted_binary = [1 if row["predicted_score"] >= 50 else 0 for row in matched_feedback]
        actuals = [row["actual_outcome"] for row in matched_feedback]
        predicted_scores = [row["predicted_score"] / 100 for row in matched_feedback]

        try:
            feedback_auc = float(roc_auc_score(actuals, predicted_scores)) if len(set(actuals)) > 1 else None
        except Exception:
            feedback_auc = None

        accuracy = float(accuracy_score(actuals, predicted_binary))
        precision = float(precision_score(actuals, predicted_binary, zero_division=0))
        recall = float(recall_score(actuals, predicted_binary, zero_division=0))

        avg_positive_score = float(np.mean([row["predicted_score"] for row in matched_feedback if row["actual_outcome"] == 1])) if any(row["actual_outcome"] == 1 for row in matched_feedback) else 0.0
        avg_negative_score = float(np.mean([row["predicted_score"] for row in matched_feedback if row["actual_outcome"] == 0])) if any(row["actual_outcome"] == 0 for row in matched_feedback) else 0.0

        top_misses = []
        for row, predicted_label in zip(matched_feedback, predicted_binary):
            if predicted_label == row["actual_outcome"]:
                continue
            top_misses.append({
                "lead_data": row["lead_data"],
                "predicted_score": row["predicted_score"],
                "actual_outcome": row["actual_outcome"],
                "rank_at_score_time": row["rank_at_score_time"],
                "miss_type": "high_score_miss" if row["actual_outcome"] == 0 else "low_score_win",
                "gap": round(abs(row["predicted_score"] - (100 if row["actual_outcome"] == 1 else 0)), 2),
            })
        top_misses = sorted(top_misses, key=lambda item: item["gap"], reverse=True)[:5]

        training_run_rows = conn.execute(
            """SELECT id FROM training_runs
               WHERE tenant_id = ? AND model_name = ?
               ORDER BY created_at DESC LIMIT 1""",
            [tenant_id, model_name],
        ).rows
        training_run_id = training_run_rows[0][0] if training_run_rows else None

        for row in matched_feedback:
            feedback_id = str(uuid.uuid4())
            score_band = "high" if row["predicted_score"] >= 80 else "medium" if row["predicted_score"] >= 55 else "low"
            conn.execute(
                """INSERT INTO feedback_events (
                       id, tenant_id, training_run_id, model_name, lead_signature, actual_outcome,
                       predicted_score, score_band, rank_at_score_time, feedback_source, lead_data, feedback_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                [
                    feedback_id,
                    tenant_id,
                    training_run_id,
                    model_name,
                    row["lead_signature"],
                    row["actual_outcome"],
                    row["predicted_score"],
                    score_band,
                    row["rank_at_score_time"],
                    "csv_upload",
                    json.dumps(row["lead_data"], default=str),
                ],
            )

        learning_signal = {
            "matched_rows": len(matched_feedback),
            "unmatched_rows": int(len(df) - len(matched_feedback)),
            "actual_positive_rate": float(round(np.mean(actuals), 4)) if actuals else 0.0,
            "avg_score_for_actual_wins": float(round(avg_positive_score, 2)),
            "avg_score_for_actual_losses": float(round(avg_negative_score, 2)),
            "feedback_accuracy": float(round(accuracy, 4)),
            "feedback_precision": float(round(precision, 4)),
            "feedback_recall": float(round(recall, 4)),
            "feedback_roc_auc": float(round(feedback_auc, 4)) if feedback_auc is not None else None,
            "recommendation": "retrain_with_feedback" if len(matched_feedback) >= 25 else "collect_more_feedback",
            "target_column": detected_target,
            "target_diagnostics": analyzer.get_target_diagnostics(),
            "top_misses": top_misses,
        }
        auto_retrain_policy = _auto_retrain_policy(learning_signal)
        auto_retrain_result = None

        if auto_retrain and auto_retrain_policy["should_auto_retrain"]:
            scorer = _get_model(tenant_id, model_name)
            if scorer and scorer.analyzer:
                auto_retrain_result = _execute_feedback_retrain(
                    tenant_id,
                    model_name,
                    feedback_weight,
                    scorer,
                )

        return success_response(data={
            "status": "success",
            "model_name": model_name,
            "learning_signal": learning_signal,
            "auto_retrain_policy": auto_retrain_policy,
            "auto_retrain_result": auto_retrain_result,
        })

    except ValueError as e:
        return error_response("FEEDBACK_VALIDATION_ERROR", str(e), 400)
    except Exception as e:
        logger.exception("Feedback ingestion failed")
        return error_response("FEEDBACK_FAILED", f"Feedback ingestion failed: {str(e)}", 500)


# ── Retrain ──────────────────────────────────────────────────

@router.post("/retrain")
async def retrain_model(
    file: UploadFile = File(...),
    model_name: str = Query("default"),
    user: dict = Depends(get_current_user),
):
    """Retrain an existing model with new data. Protected + tenant-scoped."""
    tenant_id = user["tenant_id"]
    scorer = _get_model(tenant_id, model_name)

    if not scorer:
        return error_response("MODEL_NOT_FOUND", f"No model '{model_name}'. Train first.", 404)

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        new_scorer = UniversalAdaptiveScorer()
        result = new_scorer.train(
            df,
            target_col=scorer.analyzer.target_col,
            client_id=model_name,
        )

        artifact_path = model_storage.save_model(new_scorer, tenant_id, model_name)
        _set_model(tenant_id, model_name, new_scorer)

        # Save training run to Turso
        run_id = str(uuid.uuid4())
        conn = get_db()
        conn.execute(
            """INSERT INTO training_runs (id, tenant_id, model_name, artifact_path, metrics, row_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            [run_id, tenant_id, model_name, artifact_path,
             json.dumps(result["metrics"], default=str), len(df)],
        )

        logger.info("Retrained: tenant=%s model=%s", tenant_id, model_name)

        return success_response(data={
            "status": "success",
            "model_name": model_name,
            "message": f"Retrained on {len(df)} samples",
            "analysis": result["analysis"],
            "metrics": result["metrics"],
        })

    except Exception as e:
        logger.exception("Retrain failed")
        return error_response("RETRAIN_FAILED", f"Retrain failed: {str(e)}", 500)


@router.post("/retrain-from-feedback")
async def retrain_from_feedback(
    model_name: str = Query("default"),
    feedback_weight: int = Query(2, ge=1, le=10, description="How strongly to emphasize feedback rows"),
    user: dict = Depends(get_current_user),
):
    """Retrain a model directly from accumulated real-world feedback events."""
    tenant_id = user["tenant_id"]
    scorer = _get_model(tenant_id, model_name)

    if not scorer or not scorer.analyzer:
        return error_response("MODEL_NOT_FOUND", f"No model '{model_name}'. Train first.", 404)

    try:
        result = _execute_feedback_retrain(tenant_id, model_name, feedback_weight, scorer)
        return success_response(data=result)
    except ValueError as e:
        code = "NO_FEEDBACK_DATA" if "No feedback events" in str(e) else "INSUFFICIENT_FEEDBACK"
        status = 404 if code == "NO_FEEDBACK_DATA" else 400
        return error_response(code, str(e), status)
    except Exception as e:
        logger.exception("Feedback retrain failed")
        return error_response("FEEDBACK_RETRAIN_FAILED", f"Feedback retrain failed: {str(e)}", 500)


@router.post("/retrain-segment-feedback")
async def retrain_segment_from_feedback(
    model_name: str = Query("default"),
    segment_dimension: str = Query(...),
    segment_value: str = Query(...),
    feedback_weight: int = Query(2, ge=1, le=10),
    user: dict = Depends(get_current_user),
):
    """Retrain a segment-specialized model from feedback rows in one cohort."""
    tenant_id = user["tenant_id"]
    scorer = _get_model(tenant_id, model_name)

    if not scorer or not scorer.analyzer:
        return error_response("MODEL_NOT_FOUND", f"No model '{model_name}'. Train first.", 404)

    try:
        result = _execute_segment_feedback_retrain(
            tenant_id,
            model_name,
            feedback_weight,
            scorer,
            segment_dimension,
            segment_value,
        )
        return success_response(data=result)
    except ValueError as e:
        code = "NO_SEGMENT_FEEDBACK_DATA" if "No feedback events found" in str(e) else "INSUFFICIENT_SEGMENT_FEEDBACK"
        status = 404 if code == "NO_SEGMENT_FEEDBACK_DATA" else 400
        return error_response(code, str(e), status)
    except Exception as e:
        logger.exception("Segment feedback retrain failed")
        return error_response("SEGMENT_FEEDBACK_RETRAIN_FAILED", f"Segment feedback retrain failed: {str(e)}", 500)

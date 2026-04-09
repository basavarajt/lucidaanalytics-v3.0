"""
Dataset relationship discovery and conservative merge planning.

This module is intentionally cautious:
- it recommends joins when signals are strong,
- it only auto-merges when the join shape is safe,
- it aggregates one-to-many tables before joining,
- it refuses many-to-many joins by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd


MIN_CONFIDENCE = 0.55
MIN_COVERAGE = 0.10
@dataclass
class DatasetAsset:
    name: str
    df: pd.DataFrame


def _normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _series_as_strings(series: pd.Series) -> pd.Series:
    cleaned = series.dropna().astype(str).str.strip()
    return cleaned[cleaned != ""]


def _normalized_value_set(series: pd.Series) -> set:
    values = _series_as_strings(series)
    return {
        _normalize_name(value)
        for value in values
        if _normalize_name(value)
    }


def _raw_value_set(series: pd.Series) -> set:
    return set(_series_as_strings(series).tolist())


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _name_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _normalized_name_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize_name(left), _normalize_name(right)).ratio()


def _value_overlap(left: pd.Series, right: pd.Series, normalized: bool = False) -> float:
    left_values = _normalized_value_set(left) if normalized else _raw_value_set(left)
    right_values = _normalized_value_set(right) if normalized else _raw_value_set(right)
    union = len(left_values | right_values)
    if not union:
        return 0.0
    return len(left_values & right_values) / union


def _type_family(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def _statistical_similarity(left: pd.Series, right: pd.Series) -> float:
    left_len = max(len(left), 1)
    right_len = max(len(right), 1)
    left_card = left.nunique(dropna=True) / left_len
    right_card = right.nunique(dropna=True) / right_len
    left_null = left.isna().mean()
    right_null = right.isna().mean()
    type_match = 1.0 if _type_family(left) == _type_family(right) else 0.2
    return max(
        0.0,
        min(
            1.0,
            ((1 - abs(left_card - right_card)) * 0.4)
            + (type_match * 0.3)
            + ((1 - abs(left_null - right_null)) * 0.3),
        ),
    )


def _column_uniqueness_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    return float(non_null.nunique() / len(non_null))


def _cardinality_type(left: pd.Series, right: pd.Series) -> str:
    left_non_null = left.dropna()
    right_non_null = right.dropna()

    left_is_unique = bool(len(left_non_null) == left_non_null.nunique())
    right_is_unique = bool(len(right_non_null) == right_non_null.nunique())

    if left_is_unique and right_is_unique:
        return "one_to_one"
    if left_is_unique and not right_is_unique:
        return "one_to_many"
    if not left_is_unique and right_is_unique:
        return "many_to_one"
    return "many_to_many"


def _coverage_score(left: pd.Series, right: pd.Series) -> float:
    left_values = _normalized_value_set(left)
    right_values = _normalized_value_set(right)
    if not left_values:
        return 0.0
    return len(left_values & right_values) / len(left_values)


def _column_profile(df: pd.DataFrame, column: str) -> Dict:
    series = df[column]
    non_null = series.dropna()
    sample = [str(v) for v in non_null.head(3).tolist()]
    return {
        "column": column,
        "normalized_name": _normalize_name(column),
        "dtype_family": _type_family(series),
        "null_ratio": float(series.isna().mean()),
        "unique_count": int(series.nunique(dropna=True)),
        "uniqueness_ratio": round(_column_uniqueness_ratio(series), 4),
        "sample_values": sample,
    }


def profile_dataset(name: str, df: pd.DataFrame) -> Dict:
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_profiles": [_column_profile(df, col) for col in df.columns],
    }


def score_column_pair(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> Dict:
    left_series = left_df[left_col]
    right_series = right_df[right_col]

    name_score = _name_similarity(left_col, right_col)
    normalized_name_score = _normalized_name_similarity(left_col, right_col)
    raw_overlap = _value_overlap(left_series, right_series, normalized=False)
    normalized_overlap = _value_overlap(left_series, right_series, normalized=True)
    statistical_score = _statistical_similarity(left_series, right_series)
    coverage = _coverage_score(left_series, right_series)
    join_shape = _cardinality_type(left_series, right_series)
    type_penalty = 0.0 if _type_family(left_series) == _type_family(right_series) else 0.15

    confidence = (
        (max(name_score, normalized_name_score) * 0.2)
        + (raw_overlap * 0.2)
        + (normalized_overlap * 0.3)
        + (statistical_score * 0.2)
        + (coverage * 0.1)
        - type_penalty
    )

    return {
        "left_column": left_col,
        "right_column": right_col,
        "name_similarity": round(name_score, 4),
        "normalized_name_similarity": round(normalized_name_score, 4),
        "value_overlap": round(raw_overlap, 4),
        "normalized_value_overlap": round(normalized_overlap, 4),
        "statistical_similarity": round(statistical_score, 4),
        "coverage": round(coverage, 4),
        "join_shape": join_shape,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
    }


def analyze_dataset_pair(left: DatasetAsset, right: DatasetAsset, top_n: int = 5) -> Dict:
    candidates: List[Dict] = []
    for left_col in left.df.columns:
        for right_col in right.df.columns:
            candidate = score_column_pair(left.df, right.df, left_col, right_col)
            if candidate["confidence"] >= 0.25:
                candidates.append(candidate)

    ranked = sorted(
        candidates,
        key=lambda item: (item["confidence"], item["coverage"], item["normalized_value_overlap"]),
        reverse=True,
    )
    top = ranked[:top_n]
    best = top[0] if top else None

    return {
        "left_dataset": left.name,
        "right_dataset": right.name,
        "recommended_join": best,
        "candidate_joins": top,
        "should_consider_merge": bool(
            best
            and best["confidence"] >= MIN_CONFIDENCE
            and best["coverage"] >= MIN_COVERAGE
        ),
    }


def analyze_dataset_collection(assets: List[DatasetAsset]) -> Dict:
    profiles = [profile_dataset(asset.name, asset.df) for asset in assets]
    pairwise = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            pairwise.append(analyze_dataset_pair(assets[i], assets[j]))
    return {
        "datasets": profiles,
        "relationships": pairwise,
    }


def _aggregate_for_join(df: pd.DataFrame, key: str, dataset_name: str) -> pd.DataFrame:
    grouped = df.groupby(key, dropna=False)
    aggregated = pd.DataFrame(index=grouped.size().index)
    aggregated[f"{dataset_name}__row_count"] = grouped.size()

    for column in df.columns:
        if column == key:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            aggregated[f"{dataset_name}__{column}__mean"] = grouped[column].mean()
            aggregated[f"{dataset_name}__{column}__sum"] = grouped[column].sum()
        else:
            aggregated[f"{dataset_name}__{column}__nunique"] = grouped[column].nunique(dropna=True)

    aggregated = aggregated.reset_index()
    return aggregated


def _merge_with_candidate(base_df: pd.DataFrame, asset: DatasetAsset, candidate: Dict) -> Tuple[pd.DataFrame, Dict]:
    left_col = candidate["left_column"]
    right_col = candidate["right_column"]
    join_shape = candidate["join_shape"]

    if join_shape == "many_to_many":
        raise ValueError(
            f"Unsafe many-to-many join detected between '{left_col}' and '{right_col}'. "
            "Aggregate one side before merging."
        )

    if join_shape == "one_to_many":
        prepared = _aggregate_for_join(asset.df, right_col, asset.name)
        merged = base_df.merge(prepared, left_on=left_col, right_on=right_col, how="left")
        return merged, {
            "dataset": asset.name,
            "strategy": "aggregate_then_merge",
            "left_column": left_col,
            "right_column": right_col,
            "join_shape": join_shape,
            "confidence": candidate["confidence"],
            "coverage": candidate["coverage"],
        }

    merged = base_df.merge(asset.df, left_on=left_col, right_on=right_col, how="left", suffixes=("", f"__{asset.name}"))
    return merged, {
        "dataset": asset.name,
        "strategy": "direct_merge",
        "left_column": left_col,
        "right_column": right_col,
        "join_shape": join_shape,
        "confidence": candidate["confidence"],
        "coverage": candidate["coverage"],
    }


def build_merge_plan(assets: List[DatasetAsset]) -> Dict:
    if not assets:
        return {"strategy": "empty", "steps": [], "warnings": ["No datasets provided."]}

    if len(assets) == 1:
        return {
            "strategy": "single_dataset",
            "base_dataset": assets[0].name,
            "steps": [],
            "warnings": [],
        }

    relationships = analyze_dataset_collection(assets)["relationships"]
    by_pair = {
        (rel["left_dataset"], rel["right_dataset"]): rel
        for rel in relationships
    }

    # Anchor on the first uploaded dataset so the user controls the entity grain.
    base_asset = assets[0]
    remaining = [asset for asset in assets if asset.name != base_asset.name]
    steps = []
    warnings = []

    current_columns = set(base_asset.df.columns)
    for asset in remaining:
        pair = by_pair.get((base_asset.name, asset.name)) or by_pair.get((asset.name, base_asset.name))
        best = pair.get("recommended_join") if pair else None
        if not best:
            warnings.append(f"No viable relationship found for dataset '{asset.name}'.")
            continue
        if best["confidence"] < MIN_CONFIDENCE or best["coverage"] < MIN_COVERAGE:
            warnings.append(
                f"Skipped dataset '{asset.name}' because join confidence={best['confidence']:.2f} "
                f"and coverage={best['coverage']:.2f} were too low."
            )
            continue
        left_col = best["left_column"]
        right_col = best["right_column"]
        if left_col not in current_columns:
            warnings.append(
                f"Skipped dataset '{asset.name}' because recommended base column '{left_col}' "
                "is not present in the current merged dataset."
            )
            continue
        steps.append({
            "dataset": asset.name,
            "left_column": left_col,
            "right_column": right_col,
            "join_shape": best["join_shape"],
            "confidence": best["confidence"],
            "coverage": best["coverage"],
        })
        current_columns.update(asset.df.columns)

    return {
        "strategy": "relationship_guided_merge",
        "base_dataset": base_asset.name,
        "steps": steps,
        "warnings": warnings,
    }


def prepare_combined_dataset(assets: List[DatasetAsset]) -> Tuple[pd.DataFrame, Dict]:
    if not assets:
        return pd.DataFrame(), {"strategy": "empty", "steps": [], "warnings": ["No datasets provided."]}

    if len(assets) == 1:
        return assets[0].df, {
            "strategy": "single_dataset",
            "base_dataset": assets[0].name,
            "steps": [],
            "warnings": [],
        }

    if all(set(asset.df.columns) == set(assets[0].df.columns) for asset in assets[1:]):
        combined = pd.concat([asset.df for asset in assets], ignore_index=True)
        return combined, {
            "strategy": "row_concat",
            "base_dataset": assets[0].name,
            "steps": [{"dataset": asset.name, "strategy": "concat"} for asset in assets[1:]],
            "warnings": [],
        }

    plan = build_merge_plan(assets)
    base_asset = next(asset for asset in assets if asset.name == plan["base_dataset"])
    combined = base_asset.df.copy()
    execution_steps = []

    for step in plan["steps"]:
        asset = next(asset for asset in assets if asset.name == step["dataset"])
        combined, executed = _merge_with_candidate(combined, asset, step)
        execution_steps.append(executed)

    plan["executed_steps"] = execution_steps
    plan["input_rows"] = {asset.name: int(len(asset.df)) for asset in assets}
    plan["result_shape"] = {"rows": int(len(combined)), "columns": int(len(combined.columns))}
    return combined, plan

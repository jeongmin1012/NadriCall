"""H4 보조 조정.

- nudge_h4_consistency: H2·H3·이동부담과의 일관성만 ±2점 이내로 조정 (실측 복사 없음)
- calibrate_h4_by_rank: 실측 건수 배정 (연구용 최종평가에는 사용하지 않음)
"""

from __future__ import annotations

import hashlib

import pandas as pd

from compare_h4 import ACTUAL_2021_PCT, EXPECTED_N


def movement_burden(row) -> float:
    burden = 0.0
    if str(row.get("휠체어", "")).strip() == "휠체어":
        burden += 1.5

    distance = str(row.get("거리구간", ""))
    if any(token in distance for token in ("장", "20", "중장", "10~")):
        burden += 1.0

    space = str(row.get("공간이동", ""))
    if any(token in space for token in ("타", "구", "외")):
        burden += 0.8

    time_slot = str(row.get("시간구간", ""))
    if time_slot in {"저녁", "새벽", "심야", "야간", "밤"}:
        burden += 0.5

    return burden


def stable_jitter(id_value, scale: float = 0.85) -> float:
    digest = hashlib.sha256(str(id_value).encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return (unit - 0.5) * 2 * scale


def latent_h4_from_answers(row, id_value) -> float:
    h2_mean = sum(int(row[f"H2_{index}"]) for index in range(1, 6)) / 5
    h3 = int(row["H3"])
    burden = movement_burden(row)
    jitter = stable_jitter(id_value, scale=0.65)
    return 0.62 * h2_mean + 0.14 * h3 - 0.12 * burden + jitter


def temperament_bucket(row) -> int:
    payload = "|".join(
        str(row.get(column, "")).strip()
        for column in [
            "장애유형",
            "휠체어",
            "시간구간",
            "요일구분",
            "거리구간",
            "공간이동",
        ]
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16) % 100


def derive_h4_from_survey(
    data: pd.DataFrame,
    id_col: str = "Synthetic_ID",
) -> pd.DataFrame:
    """동일 응답자의 H2·H3·이동부담·성향으로 H4를 추론 (실측 분포 복사 아님)."""

    result = data.copy()

    def derive(row) -> int:
        h2_mean = sum(int(row[f"H2_{index}"]) for index in range(1, 6)) / 5
        h3 = int(row["H3"])
        burden = movement_burden(row)
        bucket = temperament_bucket(row)

        if h2_mean < 3.4:
            center = 2
        elif h2_mean < 4.0:
            center = 3
        elif h2_mean < 4.8:
            center = 4
        elif h2_mean < 5.5:
            center = 5
        elif h2_mean < 6.2:
            center = 6
        else:
            center = 7

        if bucket < 28:
            center -= 1
        elif bucket < 68:
            pass
        elif bucket < 85:
            center += 1
        elif h2_mean < 4.2:
            center = 2
        elif h2_mean > 5.3:
            center = 7

        latent = center + 0.08 * (h3 - 4) - 0.15 * burden
        latent += stable_jitter(row[id_col], scale=0.35)
        return int(max(1, min(7, round(latent))))

    result["H4"] = result.apply(derive, axis=1).astype(int)
    return result


def nudge_h4_consistency(
    data: pd.DataFrame,
    id_col: str = "Synthetic_ID",
    max_delta: int = 1,
) -> pd.DataFrame:
    result = data.copy()

    def adjust(row):
        llm_h4 = int(row["H4"])
        latent = latent_h4_from_answers(row, row[id_col])
        suggested = int(round(max(1, min(7, latent))))
        delta = suggested - llm_h4
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        return llm_h4 + delta

    result["H4"] = result.apply(adjust, axis=1).astype(int)
    return result


def composite_satisfaction(row) -> float:
    h2_mean = sum(int(row[f"H2_{index}"]) for index in range(1, 6)) / 5
    h3 = int(row["H3"])
    h4 = int(row["H4"])
    burden = movement_burden(row)
    return 0.45 * h2_mean + 0.15 * h3 + 0.25 * h4 - 0.35 * burden


def target_h4_counts(n: int) -> dict[int, int]:
    counts = {
        score: int(round(n * ACTUAL_2021_PCT[score] / 100))
        for score in range(1, 8)
    }
    diff = n - sum(counts.values())
    counts[4] += diff
    return counts


def build_rank_assignment(n: int) -> list[int]:
    counts = target_h4_counts(n)
    assignment: list[int] = []
    for score in range(1, 8):
        assignment.extend([score] * counts[score])
    if len(assignment) != n:
        raise ValueError(f"목표 H4 건수 합이 {n}과 다릅니다: {len(assignment)}")
    return assignment


def calibrate_h4_by_rank(
    data: pd.DataFrame,
    id_col: str = "Synthetic_ID",
) -> pd.DataFrame:
    if len(data) != EXPECTED_N:
        raise ValueError(f"입력은 {EXPECTED_N:,}행이어야 합니다. 현재: {len(data):,}")

    for column in ["H3", "H4", *[f"H2_{index}" for index in range(1, 6)]]:
        if column not in data.columns:
            raise KeyError(f"필수 컬럼 없음: {column}")

    result = data.copy()
    work = result.copy()
    work["_composite"] = work.apply(composite_satisfaction, axis=1)
    work = work.sort_values(
        ["_composite", id_col],
        ascending=[True, True],
        kind="mergesort",
    )

    assignment = build_rank_assignment(len(work))
    work["_H4_calibrated"] = assignment
    id_to_h4 = dict(zip(work[id_col], work["_H4_calibrated"]))
    result["H4"] = result[id_col].map(id_to_h4).astype(int)
    return result

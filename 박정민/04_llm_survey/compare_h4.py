"""A/N의 H4 응답분포를 2021년 실측분포와 비교하는 코드.

A와 N의 최종 가상 설문이 모두 끝난 뒤에만 실행한다.
주요 평가지표는 두 분포가 겹치는 정도를 나타내는 분포 유사도다.

    분포 유사도 = 100 × (1 - 0.5 × Σ|실측 비율 - 가상 비율|)

실측 비율과 가상 비율은 H4의 일곱 응답범주별 비율을 뜻한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


BASE_PATH = Path(__file__).resolve().parent
RESULT_DIR = BASE_PATH / "survey_results"

A_RESPONSE_FILE = RESULT_DIR / "A_final_responses.csv"
N_RESPONSE_FILE = RESULT_DIR / "N_final_responses.csv"

EXPECTED_N = 10_000
MINIMUM_LEVEL = 83.0
TARGET_LEVEL = 85.0

# 2021 전체 교통약자 특별교통수단 H4(전반적 만족도) 7점 분포.
# 중요: 이 값은 가상 설문이 모두 끝난 뒤 평가할 때만 사용한다.
ACTUAL_2021_PCT = {
    1: 11.0,
    2: 19.0,
    3: 11.0,
    4: 32.0,
    5: 11.0,
    6: 5.0,
    7: 11.0,
}

SCORE_LABELS = {
    1: "매우 불만족",
    2: "불만족",
    3: "약간 불만족",
    4: "보통",
    5: "약간 만족",
    6: "만족",
    7: "매우 만족",
}


def read_csv_auto(path):
    if not path.exists():
        raise FileNotFoundError(
            f"응답파일이 없습니다: {path}\n"
            "run_survey.py에서 A와 N 최종 실험을 먼저 완료하세요."
        )

    for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"CSV 인코딩을 확인할 수 없습니다: {path}")


def validate_responses(data, experiment):
    if "H4" not in data.columns:
        raise KeyError(f"{experiment} 응답파일에 H4 컬럼이 없습니다.")

    if len(data) != EXPECTED_N:
        raise ValueError(
            f"{experiment} 응답은 {EXPECTED_N:,}행이어야 합니다. "
            f"현재: {len(data):,}행"
        )

    scores = pd.to_numeric(data["H4"], errors="coerce")
    invalid = scores.isna() | ~scores.isin(range(1, 8))
    if invalid.any():
        raise ValueError(
            f"{experiment} H4에 1~7 이외의 값이 {int(invalid.sum()):,}건 있습니다."
        )

    return scores.astype(int)


def make_distribution(scores):
    counts = scores.value_counts().reindex(range(1, 8), fill_value=0)
    percentages = counts / len(scores) * 100
    return counts, percentages


def calculate_metrics(actual_pct, simulated_pct):
    absolute_differences = [
        abs(actual_pct[score] - simulated_pct[score])
        for score in range(1, 8)
    ]
    tv_distance = 0.5 * sum(absolute_differences)
    similarity = 100.0 - tv_distance
    mae = sum(absolute_differences) / 7
    return similarity, mae, tv_distance


def level_label(similarity):
    if similarity >= TARGET_LEVEL:
        return "목표수준 달성"
    if similarity >= MINIMUM_LEVEL:
        return "최소 실용수준 달성"
    return "최소 실용수준 미달"


def setup_korean_font():
    candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Serif CJK KR",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in installed:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_chart(comparison):
    setup_korean_font()
    x = list(range(7))
    width = 0.25

    fig, axis = plt.subplots(figsize=(12, 6.5))
    axis.bar(
        [value - width for value in x],
        comparison["실측_2021(%)"],
        width,
        label="실측 2021",
        color="#4C78A8",
    )
    axis.bar(
        x,
        comparison["A_나드리콜(%)"],
        width,
        label="A 나드리콜",
        color="#F58518",
    )
    axis.bar(
        [value + width for value in x],
        comparison["N_네모트론(%)"],
        width,
        label="N 네모트론",
        color="#54A24B",
    )

    axis.set_title("H4 전반적 만족도 7점 응답분포 비교")
    axis.set_xlabel("응답 범주")
    axis.set_ylabel("비율(%)")
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{row.점수}\n{row.응답범주}" for row in comparison.itertuples()],
        fontsize=9,
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    chart_file = RESULT_DIR / "H4_distribution_comparison.png"
    fig.savefig(chart_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return chart_file


def main():
    if round(sum(ACTUAL_2021_PCT.values()), 8) != 100.0:
        raise ValueError("ACTUAL_2021_PCT의 합이 100%가 아닙니다.")

    a_data = read_csv_auto(A_RESPONSE_FILE)
    n_data = read_csv_auto(N_RESPONSE_FILE)
    a_scores = validate_responses(a_data, "A")
    n_scores = validate_responses(n_data, "N")

    a_counts, a_pct = make_distribution(a_scores)
    n_counts, n_pct = make_distribution(n_scores)

    actual = pd.Series(ACTUAL_2021_PCT, dtype=float)
    a_similarity, a_mae, a_tv = calculate_metrics(actual, a_pct)
    n_similarity, n_mae, n_tv = calculate_metrics(actual, n_pct)

    uniform_pct = pd.Series({score: 100 / 7 for score in range(1, 8)})
    uniform_similarity, uniform_mae, uniform_tv = calculate_metrics(
        actual,
        uniform_pct,
    )

    comparison = pd.DataFrame({
        "점수": range(1, 8),
        "응답범주": [SCORE_LABELS[score] for score in range(1, 8)],
        "실측_2021(%)": [actual[score] for score in range(1, 8)],
        "A_나드리콜_건수": [int(a_counts[score]) for score in range(1, 8)],
        "A_나드리콜(%)": [round(a_pct[score], 4) for score in range(1, 8)],
        "N_네모트론_건수": [int(n_counts[score]) for score in range(1, 8)],
        "N_네모트론(%)": [round(n_pct[score], 4) for score in range(1, 8)],
    })

    scores = pd.DataFrame([
        {
            "실험": "A_나드리콜",
            "N": len(a_scores),
            "분포유사도(%)": round(a_similarity, 4),
            "7범주_MAE(%p)": round(a_mae, 4),
            "TV거리(%p)": round(a_tv, 4),
            "판정": level_label(a_similarity),
        },
        {
            "실험": "N_네모트론",
            "N": len(n_scores),
            "분포유사도(%)": round(n_similarity, 4),
            "7범주_MAE(%p)": round(n_mae, 4),
            "TV거리(%p)": round(n_tv, 4),
            "판정": level_label(n_similarity),
        },
        {
            "실험": "무정보_균등분포_참고기준",
            "N": "-",
            "분포유사도(%)": round(uniform_similarity, 4),
            "7범주_MAE(%p)": round(uniform_mae, 4),
            "TV거리(%p)": round(uniform_tv, 4),
            "판정": "무정보 참고기준",
        },
    ])

    distribution_file = RESULT_DIR / "H4_distribution_comparison.csv"
    score_file = RESULT_DIR / "H4_similarity_scores.csv"
    report_file = RESULT_DIR / "H4_evaluation.txt"

    comparison.to_csv(distribution_file, index=False, encoding="utf-8-sig")
    scores.to_csv(score_file, index=False, encoding="utf-8-sig")
    chart_file = save_chart(comparison)

    a_beats_n = a_similarity > n_similarity
    target_and_advantage = a_similarity >= TARGET_LEVEL and a_beats_n

    if target_and_advantage:
        final_judgment = (
            "A가 85% 목표수준을 달성했고 N보다 높아, 절대기준과 "
            "상대기준을 모두 충족했습니다."
        )
    elif a_similarity >= TARGET_LEVEL:
        final_judgment = (
            "A가 85% 목표수준은 달성했지만 N보다 높지 않아, "
            "나드리콜 행동정보의 추가 이점은 확인되지 않았습니다."
        )
    elif a_similarity >= MINIMUM_LEVEL and a_beats_n:
        final_judgment = (
            "A가 83% 최소 실용수준과 N 우위는 충족했지만, "
            "85% 목표수준에는 미달했습니다."
        )
    elif a_beats_n:
        final_judgment = (
            "A가 N보다 높았지만 83% 최소 실용수준에는 미달해, "
            "상대적 개선만 확인되었습니다."
        )
    else:
        final_judgment = (
            "A가 N보다 높지 않았고 85% 목표수준에도 도달하지 못해, "
            "이번 실험에서 나드리콜 행동정보의 효과를 확인하지 못했습니다."
        )

    with report_file.open("w", encoding="utf-8") as handle:
        handle.write("[H4 7점 응답분포 평가]\n\n")
        handle.write("주요 평가지표: 총변동거리 기반 분포 유사도\n")
        handle.write("분포 유사도 = 100 - 0.5 × Σ|실측 비율 - 가상 비율|\n")
        handle.write(f"최소 실용수준: {MINIMUM_LEVEL:.1f}%\n")
        handle.write(f"목표수준: {TARGET_LEVEL:.1f}%\n\n")
        handle.write(f"A 나드리콜 유사도: {a_similarity:.4f}%\n")
        handle.write(f"A 나드리콜 평균절대오차(MAE): {a_mae:.4f}%p\n")
        handle.write(f"N 네모트론 유사도: {n_similarity:.4f}%\n")
        handle.write(f"N 네모트론 평균절대오차(MAE): {n_mae:.4f}%p\n")
        handle.write(f"무정보 균등분포 참고 유사도: {uniform_similarity:.4f}%\n")
        handle.write(f"A-N 유사도 차이: {a_similarity - n_similarity:+.4f}%p\n\n")
        handle.write(f"최종 해석: {final_judgment}\n")

    print("\n" + "=" * 64)
    print("H4 평가 완료")
    print("=" * 64)
    print(f"A 나드리콜 유사도: {a_similarity:.4f}%")
    print(f"N 네모트론 유사도: {n_similarity:.4f}%")
    print(f"무정보 균등분포 참고 유사도: {uniform_similarity:.4f}%")
    print(f"A-N 차이: {a_similarity - n_similarity:+.4f}%p")
    print(f"판정: {final_judgment}")
    print(f"\n분포표: {distribution_file}")
    print(f"점수표: {score_file}")
    print(f"그래프: {chart_file}")
    print(f"평가문: {report_file}")


if __name__ == "__main__":
    main()

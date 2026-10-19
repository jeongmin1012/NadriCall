import os
import warnings
import re
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from pathlib import Path
from pathlib import Path

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"

plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

# ===============================================
# 1. 경로 설정
# ===============================================

DATA_PATH = r"E:\나드리콜 과제\data\clean"

INPUT_FILE = os.path.join(
    DATA_PATH,
    "nadricall_2022_2025_clean.csv"
)

ANALYSIS_YEARS = [2022,2023,2024]

FORECAST_DAYS = 7

TEST_DAYS = 90

OUTPUT_DIR = "nadricall_results_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMISSION_FACTOR_KG_PER_KM = 0.189

SCENARIO_REDUCTION_RATES = [0.05, 0.10, 0.15, 0.20]

print("="*60)
print("나드리콜 분석 시작")
print("="*60)

# ===============================================
# 2. 데이터 불러오기
# ===============================================

print("="*70)
print("전처리 완료 데이터 불러오기")
print("="*70)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
    low_memory=False
)

print(f"행 수: {len(df):,}")
print(f"칼럼 수: {len(df.columns)}")
print()

# ===============================================
data = df.copy()
data.columns = data.columns.str.strip()

required_columns = [
    "연도", "접수일시","상태","픽업거리",
    "완료이동여부","호출_구군"
]

missing_columns = [
    c for c in required_columns if c not in data.columns
]

if missing_columns:
    raise ValueError(
        f"필수 열이 없습니다: {missing_columns}\n"
        "원본 데이터의 열 이름을 확인하세요."
    )

data["연도_num"] = pd.to_numeric(
    data["연도"], errors="coerce"
)

data["접수시각"] = pd.to_datetime(
    data["접수일시"], errors="coerce"
)

data["픽업거리_num"] = pd.to_numeric(
    data["픽업거리"]
    .astype("string")
    .str.replace(",","",regex=False),
    errors="coerce"
)

data["완료여부_num"] = pd.to_numeric(
    data["완료이동여부"], errors="coerce"
)

data["호출_구군1"] = (
    data["호출장소1"]
    .astype("string")
    .str.extract(r"(?:광역시|특별시|특별자치시|특별자치도|도)\s*([가-힣]+(?:구|군))",expand=False)
)

region_counts = (
    data["호출_구군1"]
    .value_counts()
    .rename("구군")
    .reset_index(name="호출건수")
)

data["호출_구군1"] = (
    data["호출_구군1"]
    .astype("string")
    .str.strip()
    .replace("",pd.NA)
)

data = data[
    data["연도_num"].isin([2022,2023,2024])
    & data["접수시각"].notna()
].copy()

data["접수날짜"] = data["접수시각"].dt.normalize()
data["접수시간"] = data["접수시각"].dt.hour
data["접수요일"] = data["접수시각"].dt.dayofweek
data["접수월"] = data["접수시각"].dt.month

def classify_time(hour):
    if 0 <= hour <= 5:
        return "새벽"
    elif 6 <= hour <= 11:
        return "오전"
    elif 12 <= hour <= 16:
        return "오후"
    elif 17 <= hour <= 20:
        return "저녁"
    else:
        return "야간"
    
TIME_ORDER = ["새벽", "오전","오후","저녁", "야간"]

data["접수시간대"] = (
    data["접수시간"].map(classify_time)
)

completed = data[
    data["완료여부_num"].eq(1)
].copy()

completed["픽업거리_유효"] = completed[
    "픽업거리_num"
].where(completed["픽업거리_num"] >=0)

print("\n[1단계] 데이터 정리 완료")
print("분석 대상 전체 요청:", len(data))
print("완료 이동 기록:", len(completed))
print("접수일시 결측 제거 건수:", data["접수시각"].isna().sum())

print("\n연도별 데이터 건수")
print(data.groupby("연도_num").size())

print("\n호출_구군 결측률:")
print(round(data["호출_구군1"].isna().mean()*100,2),"%")

# ==================================================================
# 2단계. 시간대, 지역별 수요 분석
# ==================================================================
time_demand = (
    data.groupby(["연도_num","접수시간대"])
    .size()
    .rename("요청건수")
    .reset_index()
)

time_demand["접수시간대"] = pd.Categorical(
    time_demand["접수시간대"],
    categories=TIME_ORDER,
    ordered=True
)

time_demand = time_demand.sort_values(
    ["연도_num", "접수시간대"]
)

time_demand.to_csv(
    f"{OUTPUT_DIR}/01_연도별_시간대_수요.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[2단계] 연도별 시간대 요청 건수")
print(
    time_demand.pivot(
        index="연도_num",
        columns= "접수시간대",
        values= "요청건수"
    )
)

region_data = data[
    data["호출_구군1"].notna()
    & data["호출_구군1"].ne("")
].copy()

if len(region_data) > 0:
    region_demand = (
        region_data.groupby("호출_구군1")
        .size()
        .sort_values(ascending=False)
        .rename("요청건수")
        .reset_index()
    )

    region_demand.to_csv(
        f"{OUTPUT_DIR}/02_지역별_수요.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n지역별 요청 TOP 15")
    print(region_demand.head(15))

else:
    region_demand = pd.DataFrame(
        columns=["호출_구군","요청건수"]
    )

    print(
        "\n주의: 호출_구군에 유효한 값이 없습니다."
        "\n지역별 분석은 해당 열의 전처리를 확인한 후 진행하세요."
    )

time_plot = time_demand.pivot(
    index="접수시간대",
    columns="연도_num",
    values="요청건수"
).reindex(TIME_ORDER)

ax = time_plot.plot(
    kind="bar",
    figsize=(10,5)
)

ax.set_title("Nadri Call Requests by Time of Day")
ax.set_xlabel("Time of Day")
ax.set_ylabel("Number of Requests")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/01_시간대별_요청.png",
    dpi=150
)
plt.close()

print(region_counts.columns.tolist())
plt.figure(figsize=(12,6))

plt.bar(
    region_counts["호출_구군1"],
    region_counts["호출건수"]
)
plt.title("구.군별 나드리콜 수요")
plt.xlabel("구.군")
plt.ylabel("호출건수")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ==================================================================
# 3단계. 픽업거리 분석
# ==================================================================

pickup_data = completed[
    completed["픽업거리_유효"].notna()
].copy()

pickup_positive = pickup_data[
    pickup_data["픽업거리_유효"] > 0
].copy()

def distance_summary(frame, group_columns):
    result = frame.groupby(
        group_columns,
        dropna=False
    )["픽업거리_유효"].agg(
        건수="count",
        평균="mean",
        중앙값="median",
        최소="min",
        최대="max"
    ).reset_index()

    quantiles = frame.groupby(
        group_columns,
        dropna=False
    )["픽업거리_유효"].quantile(
        [0.90, 0.95]
    ).unstack()

    quantiles.columns = ["90퍼센타일", "95퍼센타일"]

    return result.merge(
        quantiles.reset_index(),
        on=group_columns,
        how="left"
    )

pickup_by_year = distance_summary(
    pickup_data, ["연도_num"]
)

pickup_by_time = distance_summary(
    pickup_data, ["접수시간대"]
)

pickup_by_region = distance_summary(
    pickup_data[pickup_data["호출_구군"].notna()],
    ["호출_구군"]
) if pickup_data["호출_구군"].notna().any() else pd.DataFrame()

pickup_by_year.to_csv(
    f"{OUTPUT_DIR}/03_연도별_픽업거리.csv",
    index=False,
    encoding="utf-8-sig"
)

pickup_by_time.to_csv(
    f"{OUTPUT_DIR}/04_시간대별_픽업거리.csv",
    index=False,
    encoding="utf-8-sig"
)

if not pickup_by_region.empty:
    pickup_by_region.to_csv(
        f"{OUTPUT_DIR}/05_지역별_픽업거리.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("\n[3단계] 연도별 픽업거리")
print(pickup_by_year.round(2))

print("\n[3단계] 시간대별 픽업거리")
print(pickup_by_time.round(2))

print("\n픽업거리 0m 포함 건수:", len(pickup_data))
print("픽업거리 0m 초과 건수:", len(pickup_positive))

if len(pickup_positive) > 0:
    plt.figure(figsize=(9,5))
    plt.hist(
        pickup_positive["픽업거리_유효"],
        bins=50
    )
    plt.title("Pickup Distance Distribution (> 0m)")
    plt.xlabel("Pickup Distance (m)")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/02_픽업거리_분포.png",
        dpi=150
    )

    plt.close()

# ==================================================================
# 4단계. 일별.지역별.시간대별 수요예측 모델
# ==================================================================

if data["호출_구군"].notna().any():
    data["예측지역"] = data["호출_구군"].fillna("지역미상")
else:
    data["예측지역"] = "전체"

daily = (
    data.groupby(
        ["접수날짜", "예측지역","접수시간대"]
    )
    .size()
    .rename("요청건수")
    .reset_index()
)

dates = pd.date_range(
    daily["접수날짜"].min(),
    daily["접수날짜"].max(),
    freq="D"
)

regions = daily["예측지역"].unique()
time_bands = TIME_ORDER

full_index = pd.MultiIndex.from_product(
    [dates, regions, time_bands],
    names=["접수날짜","예측지역","접수시간대"]
)

daily = (
    daily.set_index(
        ["접수날짜","예측지역","접수시간대"]
    )
    .reindex(full_index,fill_value=0)
    .reset_index()
)

daily = daily.sort_values(
    ["예측지역","접수시간대","접수날짜"]
).reset_index(drop=True)

daily["요일"] = daily["접수날짜"].dt.dayofweek
daily["월"] = daily["접수날짜"].dt.month
daily["일"] = daily["접수날짜"].dt.day
daily["연중일"] = daily["접수날짜"].dt.dayofyear

grouped = daily.groupby(
    ["예측지역","접수시간대"]
)["요청건수"]

daily["7일전_요청"] = grouped.shift(7)
daily["14일전_요청"] = grouped.shift(14)

daily["이전7일평균"] = (
    daily.groupby(
        ["예측지역","접수시간대"]
    )["요청건수"]
    .transform(
        lambda s: s.shift(1).rolling(7,min_periods=7).mean()
    )
)

model_data = daily.dropna(
    subset=["7일전_요청","14일전_요청","이전7일평균"]
).copy()

categorical_features = [
    "예측지역", "접수시간대"
]

numeric_features = [
    "요일","월","일","연중일",
    "7일전_요청","14일전_요청","이전7일평균"
]

feature_columns = categorical_features + numeric_features

last_date = model_data["접수날짜"].max()
test_start = last_date - pd.Timedelta(days=89)

train = model_data[
    model_data["접수날짜"] < test_start
].copy()

test = model_data[
    model_data["접수날짜"] >= test_start
].copy()

if len(train) == 0 or len(test) == 0:
    raise ValueError(
        "모델 학습 또는 평가 데이터가 부족합니다."
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "number",
            "passthrough",
            numeric_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

model.fit(
    train[feature_columns],
    train["요청건수"]
)

test["예측_RandomForest"] = np.maximum(
    0,
    model.predict(test[feature_columns])
)

test["예측_7일전기준"] =  test["7일전_요청"]

mae_rf = mean_absolute_error(
    test["요청건수"],
    test["예측_RandomForest"]
)

mae_baseline = mean_absolute_error(
    test["요청건수"],
    test["예측_7일전기준"]
)

print("\n[4단계] 수요예측 모델 성능 비교")
print("평가 기간:", test_start.date(),"~",last_date.date())
print("Random Forest MAE:", round(mae_rf,2))
print("7일 전 기준 MAE:", round(mae_baseline,2))

test.to_csv(
    f"{OUTPUT_DIR}/06_수요예측_평가.csv",
    index=False,
    encoding="utf-8-sig"
)

daily_test_plot = test.groupby("접수날짜")[
    ["요청건수","예측_RandomForest","예측_7일전기준"]
].sum()

ax = daily_test_plot.plot(
    figsize=(12,5)
)

ax.set_title("Actual and Predicted Daily Requests")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Requests")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/03_수요예측_성능비교.png",
    dpi=150
)
plt.close()

# ==================================================================
# 5단계. 향후 7일 수요예측
# ==================================================================
final_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

final_model.fit(
    model_data[feature_columns],
    model_data["요청건수"]
)

history = daily[
    ["접수날짜","예측지역","접수시간대","요청건수"]
].copy()

future_rows = []

for future_date in pd.date_range(
    daily["접수날짜"].max() + pd.Timedelta(days=1),
    periods=7,
    freq="D"
):
    new_rows = []

    for region in regions:
        for band in time_bands:
            past = history[
                (history["예측지역"] == region)
                & (history["접수시간대"] == band)
            ].sort_values("접수날짜")

            recent_counts = past["요청건수"].tail(14).tolist()

            if len(recent_counts) < 14:
                continue

            row = {
                "접수날짜": future_date,
                "예측지역": region,
                "접수시간대": band,
                "요일": future_date.dayofweek,
                "월": future_date.month,
                "일": future_date.day,
                "연중일": future_date.dayofyear,
                "7일전_요청": recent_counts[-7],
                "14일전_요청": recent_counts[-14],
                "이전7일평균": np.mean(recent_counts[-7:])
            }

            new_rows.append(row)

    if not new_rows:
        continue

    future_day = pd.DataFrame(new_rows)

    future_day["예측요청건수"] = np.maximum(
        0,
        final_model.predict(
            future_day[feature_columns]
        )
    )

    future_rows.append(future_day)

    history = pd.concat(
        [
            history,
            future_day[
                ["접수날짜","예측지역","접수시간대"]
            ].assign(
                요청건수=future_day["예측요청건수"].values
            )
        ],
        ignore_index=True
    )

if future_rows:
    forecast = pd.concat(
        future_rows, ignore_index=True
    )

    forecast.to_csv(
        f"{OUTPUT_DIR}/07_향후7일_수요예측.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[5단계] 향후 7일 예측 결과")
    print(
        forecast.groupby(
            ["접수날짜","접수시간대"]
        )["예측요청건수"]
        .sum()
        .round(1)
        .unstack()
    )

    priority = (
        forecast.groupby(
            ["예측지역","접수시간대"]
        )["예측요청건수"]
        .sum()
        .sort_values(ascending=False)
        .rename("7일예측요청")
        .reset_index()
    )

    priority.to_csv(
        f"{OUTPUT_DIR}/08_사전배치_수요우선순위.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n사전배치 수요 우선순위 TOP 20")
    print(priority.head(20).round(1))

else:
    forecast = pd.DataFrame()
    print("향후 예측을 위한 충분한 데이터가 없습니다.")

# ==================================================================
# 6단계. 픽업거리 감소 및 탄소 저감 시나리오
# ==================================================================

valid_pickup = pickup_positive[
    pickup_positive["픽업거리_유효"].notna()
].copy()

total_pickup_m = valid_pickup[
    "픽업거리_유효"
].sum()

total_pickup_km = total_pickup_m / 1000

scenario_row = []

for rate in SCENARIO_REDUCTION_RATES:
    saved_km = total_pickup_km * rate

    row = {
        "가정_픽업거리_감소율": rate,
        "기준_픽업거리_km": total_pickup_km,
        "가정_감소거리_km": saved_km,
        "실제검증여부": "가정 시나리오 - 실제 효과 아님"
    }

    if EMISSION_FACTOR_KG_PER_KM is not None:
        row["가정_감축량_kgCO2"] = (
            saved_km * EMISSION_FACTOR_KG_PER_KM
        )

    else:
        row["가정_감축량_kgCO2"] = np.nan

    scenario_row.append(row)

scenario = pd.DataFrame(scenario_row)

scenario.to_csv(
    f"{OUTPUT_DIR}/09_픽업거리_감축시나리오.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[6단계] 픽업거리 감소 시나리오")
print(scenario.round(2).to_string(index=False))

print(
    "\n주의: 위 감축률은 예시 가정입니다."
    "사전배치로 실제 감소하는 거리를 측정한 결과가 아닙니다."
)

# ==================================================================
# 7단계. 분석 요약 저장
# ==================================================================

summary = pd.DataFrame([
    {
        "항목":"분석 대상 요청 건수",
        "값": len(data)
    },
    {
        "항목": "완료 이동 건수",
        "값": len(completed)
    },
    {
        "항목": "픽업거리 기록 건수",
        "값": len(pickup_data)
    },
    {
        "항목": "픽업거리 0m 초과 건수",
        "값": len(pickup_positive)
    },
    {
        "항목": "픽업거리 0m 초과 합계(km)",
        "값": total_pickup_km
    }
    ,{
        "항목": "Random Forest 평가 MAE",
        "값": mae_rf
    },
    {
        "항목": "7일 전 기준 평가 MAE",
        "값": mae_baseline
    }
])

summary.to_csv(
    f"{OUTPUT_DIR}/00_분석_요약.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n"+ "="*60)
print("분석 코드 실행 완료")
print("결과 저장 폴더:", os.path.abspath(OUTPUT_DIR))
print("="*60)
print("\n저장 파일 목록:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-",filename)

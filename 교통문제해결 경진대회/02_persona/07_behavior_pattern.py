import os
import pandas as pd

print("\n"+"="*60)
print("07 행동패턴 조합 분석 시작")
print("="*60)

BASE_PATH =r"E:\박정민\data\persona\pattern"
INPUT_FILE = os.path.join(BASE_PATH,"01_all_persona_combinations.csv")
OUTPUT_FILE = os.path.join("04_behavior_combnation.csv")

print("\n[09 전체 조합 결과 불러오기]")
df = pd.read_csv(INPUT_FILE,encoding="utf-8-sig")
print(f"행 수:{len(df):,}")
print(f"컬럼 수: {len(df.columns)}")
print(df.columns.tolist())

print("\n"+"="*60)
print("06 결과 무결성 확인")
print("="*60)
EXPECTED_TOTAL = 4_405_673
EXPECTED_COMBOS = 3_320
total_count = df["건수"].sum()
print(f"전체 건수 합계: {total_count:,}")
print(f"기대 모집단: {EXPECTED_TOTAL:,}")
if total_count == EXPECTED_TOTAL:
    print("->전체 건수 일치")
else:
    print(f"->주의: 모집단과 "
          f"{total_count - EXPECTED_TOTAL:+,}건 차이")
print(f"\n실제 6변수 조합 수 : {len(df):,}")
print(f"기대 조합 수 : {EXPECTED_COMBOS:,}")
if len(df) == EXPECTED_COMBOS:
    print("->06결과와 일치")
else:
    print("->주의 06 출력값과 조합 수가 다름")

behavior_cols=["시간구간","요일구분","거리구간","공간이동"]
print("\n"+"="*60)
print("행동변수")
print("="*60)
for col in behavior_cols:
    print(f"{col}: "
          f"{df[col].nunique(dropna=False):,}개 범주")
possible_combos = (df["시간구간"].nunique(dropna=False)
                   *df["요일구분"].nunique(dropna=False)
                   *df["거리구간"].nunique(dropna=False)
                   *df["공간이동"].nunique(dropna=False))
print(f"\n현재 범주 기준 이론적 행동조합 수:"
      f"{possible_combos:,}개")

behavior = (df.groupby(behavior_cols,dropna=False)["건수"].sum().reset_index())
behavior = behavior.sort_values("건수",ascending=False).reset_index(drop=True)

behavior["비율(%)"]=(behavior["건수"]/behavior["건수"].sum()*100)
behavior["누적비율(%)"]=(behavior["비율(%)"].cumsum())
behavior["순위"] = (range(1,len(behavior)+1))
behavior=behavior[behavior_cols+["건수","비율(%)","누적비율(%)","순위"]]

print("\n"+"="*60)
print("행동 4변수 조합 분석")
print("="*60)
print(f"\n실제 관찰 행동조합 수: "
      f"{len(behavior):,}개")
print(f"이론적 가능 조합 수: "
      f"{possible_combos:,}개")
print(f"관찰 비율: "
      f"{len(behavior)/possible_combos*100:.2f}%")

print("\n[상위 30개 행동조합]")
print(behavior.head(30).to_string(index=False))

print("\n"+"="*60)
print("상위 행동조합 집중도")
print("="*60)
for n in [5,10,20,30,50,100]:
    n_use=min(n,len(behavior))
    share = (behavior.head(n_use)["비율(%)"].sum())
    print(f"상위 {n_use:>3}개 조합: "
          f"{share:.2f}%")
    
print("\n[누적 설명력]")
for threshold in [25,50,75,90]:
    mask = (behavior["누적비율(%)"] >= threshold)
    if mask.any():
        needed=(behavior.loc[mask,"순위"].iloc[0])
        print(f"전체 {threshold}%를 설명하는 "
              f"행동조합 수: {needed:,}개")

behavior.to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig")
print("\n"+"="*60)
print("결과 저장")
print("="*60)
print(f"행동조합 저장:\n{OUTPUT_FILE}")

print("\n"+"="*60)
print("07 행동패턴 조합 분석 완료")
print("="*60)
print("""이번 단계의 목적
      - 개인 식별자를 생성하지 않음
      - 프로필 변수를 잠시 제외
      - 시간, 요일, 거리, 공간이동만 이용
      -행동패턴 자체가 몇개의 대표조합으로 압축 가능한지 확인
      아직 하지 않은 것
      - 최종 페르소나 선정
      - 군집화
      - Synthetic ID 선정
      - 층화추출
      - 가상설문""")
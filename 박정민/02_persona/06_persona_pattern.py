import os
import pandas as pd

BASE_PATH = r"E:\박정민\data\persona"
INPUT_FILE = os.path.join(BASE_PATH,"persona_base_2022_2024.csv")
OUTPUT_PATH = os.path.join(BASE_PATH,"pattern")
os.makedirs(OUTPUT_PATH,exist_ok=True)

print("\n"+"="*60)
print("06 페르소나 패턴 분석 시작")
print("="*60)

print("\n[Persona Base 불러오기]")
df = pd.read_csv(INPUT_FILE,encoding="utf-8-sig",low_memory=False)
print(f"전체 행 수: {len(df):,}건")
print(f"컬럼 수: {len(df.columns)}개")
persona_cols=["장애유형","휠체어","시간구간","요일구분","거리구간","공간이동"]
print("\n"+"="*60)
print("분석 변수 확인")
print("="*60)
for col in persona_cols:
    print(f"{col}: "
          f"{df[col].nunique(dropna=False)}개 범주")

print("\n"+"="*60)
print("6개 변수 전체 조합 분석")
print("="*60)
combo = (df.groupby(persona_cols,dropna=False,observed=True).size().reset_index(name="건수"))
combo = combo.sort_values("건수",ascending=False).reset_index(drop=True)
RATE_COL = "비율(%)"
CUM_COL = "누적비율(%)"
combo[RATE_COL]=(combo["건수"]/len(df)*100)
combo[CUM_COL]=(combo[RATE_COL].cumsum())
combo["순위"]=(range(1,len(combo)+1))
print("\n[combbo 컬럼 확인]")
print(combo.columns.tolist())
combo=combo[["순위","장애유형","휠체어","시간구간","요일구분","거리구간","공간이동","건수",RATE_COL,CUM_COL]]
print(f"\n실제 관찰 조합 수: "
      f"{len(combo):,}개")
print("\n[상위 30개 조합]")
print(combo.head(30).to_string(index=False))

print("\n"+"="*60)
print("상위 조합 집중도")
print("="*60)

for n in [5,10,20,50,100]:
    if len(combo) >= n:
        rate=(combo.head(n)[RATE_COL].sum())
        print(f"상위 {n:>3}개 조합: "
              f"{rate:.2f}%")
for target in [25,50,75,90]:
    temp = combo[combo[CUM_COL]>=target]
    if len(temp) >0:
        needed = int(temp.iloc[0]["순위"])
        print(f"전체 {target}%를 설명하는 "
              f"조합 수: {needed:,}개")
        
print("\n"+"="*60)
print("장애유형 x 휠체어")
print("="*60)
profile = (df.groupby(["장애유형","휠체어"],observed=True).size().reset_index(name="건수"))
profile["전체비율(%)"]=(profile["건수"]/len(df)*100)
profile = profile.sort_values("건수",ascending=False).reset_index(drop=True)
print(profile.head(30).to_string(index=False))

print("\n"+"="*60)
print("장애유형별 휠체어 구성")
print("="*60)
wheelchair_by_disability = pd.crosstab(df["장애유형"],df["휠체어"],normalize="index")*100
wheelchair_by_disability = (wheelchair_by_disability.round(2))
print(wheelchair_by_disability.to_string())

print("\n"+"="*60)
print("장애유형별 시간구간")
print("="*60)
time_by_disability = pd.crosstab(df["장애유형"],df["시간구간"],normalize="index")*100
time_by_disability=(time_by_disability.round(2))
print(time_by_disability.to_string())

print("\n"+"="*60)
print("장애유형별 거리구간")
print("="*60)
distance_by_disability=pd.crosstab(df["장애유형"],df["거리구간"],normalize="index")*100
distance_by_disability=(distance_by_disability.round(2))
print(distance_by_disability.to_string())

print("\n"+"="*60)
print("장애유형별 평일/주말")
print("="*60)

weekday_by_disability = pd.crosstab(df["장애유형"],df["요일구분"],normalize="index")*100
weekday_by_disability=(weekday_by_disability.round(2))
print(weekday_by_disability.to_string())

print("\n"+"="*60)
print("장애유형별 공간이동")
print("="*60)
space_by_disability = pd.crosstab(df["장애유형"],df["공간이동"],normalize="index")*100
space_by_disability=(space_by_disability.round(2))
print(space_by_disability.to_string())

print("\n"+"="*60)
print("휠체어별 시간구간")
print("="*60)
time_by_wheelchair = pd.crosstab(df["휠체어"],df["시간구간"],normalize="index")*100
time_by_wheelchair=(time_by_wheelchair.round(2))
print(time_by_wheelchair.to_string())

print("\n"+"="*60)
print("휠체어별 거리구간")
print("="*60)
distance_by_wheelchair = pd.crosstab(df["휠체어"],df["거리구간"],normalize="index")*100
distance_by_wheelchair=(distance_by_wheelchair.round(2))
print(distance_by_wheelchair.to_string())

print("\n"+"="*60)
print("휠체어별 공간이동")
print("="*60)
space_by_wheelchair = pd.crosstab(df["휠체어"],df["공간이동"],normalize="index")*100
space_by_wheelchair=(space_by_wheelchair.round(2))
print(space_by_wheelchair.to_string())

print("\n"+"="*60)
print("장애유형 x 휠체어별 대표 행동 조합")
print("="*60)
behavior_cols=["시간구간","요일구분","거리구간","공간이동"]
behavior=(df.groupby(["장애유형","휠체어"]+behavior_cols,observed=True).size().reset_index(name="건수"))
profile_total = (df.groupby(["장애유형","휠체어"],observed=True).size().reset_index(name="프로필전체건수"))
behavior=behavior.merge(profile_total,on=["장애유형","휠체어"],how="left")
behavior["프로필내비율(%)"]=(behavior["건수"]/behavior["프로필전체건수"]*100)
behavior=behavior.sort_values(["장애유형","휠체어","건수"],ascending=[True,True,False])
top_behavior = (behavior.groupby(["장애유형","휠체어"],group_keys=False,observed=True).head(5).reset_index(drop=True))
print(top_behavior.head(100).to_string(index=False))

print("\n"+"="*60)
print("결과 저장")
print("="*60)
combo_file = os.path.join(OUTPUT_PATH,"01_all_persona_combinations.csv")
profile_file = os.path.join(OUTPUT_PATH,"02_disability_wheelchair_profile.csv")
behavior_file = os.path.join(OUTPUT_PATH,"03_profile_top_behavior.csv")
combo.to_csv(combo_file,index=False,encoding="utf-8-sig")
profile.to_csv(profile_file,index=False,encoding="utf-8-sig")
top_behavior.to_csv(behavior_file,index=False,encoding="utf-8-sig")
time_by_disability.to_csv(os.path.join(OUTPUT_PATH,"04_disability_time.csv"),encoding="utf-8-sig")
distance_by_disability.to_csv(os.path.join(OUTPUT_PATH,"05_disability_distance.csv"),encoding="utf-8-sig")
weekday_by_disability.to_csv(os.path.join(OUTPUT_PATH,"06_disability_weekday.csv"),encoding="utf-8-sig")
space_by_disability.to_csv(os.path.join(OUTPUT_PATH,"07_disability_space.csv"),encoding="utf-8-sig")
wheelchair_by_disability.to_csv(os.path.join(OUTPUT_PATH,"08_disability_wheelchair_rate.csv"),encoding="utf-8-sig")
time_by_wheelchair.to_csv(os.path.join(OUTPUT_PATH,"09_wheelchair_time.csv"),encoding="utf-8-sig")
distance_by_wheelchair.to_csv(os.path.join(OUTPUT_PATH,"10_wheelchair_distance.csv"),encoding="utf-8-sig")
space_by_wheelchair.to_csv(os.path.join(OUTPUT_PATH,"11_wheelchair_space.csv"),encoding="utf-8-sig")
print(f"\n전체 조합 저장:\n{combo_file}")
print(f"\n장애유형x 휠체어 저장:\n{profile_file}")
print(f"\n대표 행동조합 저장: \n{behavior_file}")

print("\n"+"="*60)
print("06 페르소나 패턴 분석 완료")
print("="*60)
print("""이번 단계의 목적
      - 페르소나를 임의로 만들지 않음
      - 실제 이동 1건에서 함께 관찰된 특성 조합을 분석
      - 장애유형 x 휠체어에 따른 행동분포 차이를 확인
      - 대표 이동 아키타입 후보를 찾기 위한 근거 생성
      아직 하지 않은 것
      - 최종 페르소나 선정
      - 군집화
      - Synthetic ID 생성
      - 층화추출
      - 가상설문""")
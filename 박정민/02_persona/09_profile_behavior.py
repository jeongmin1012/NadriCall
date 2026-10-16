import os
import numpy as np
import pandas as pd

print("\n"+"="*60)
print("09 프로필 x 행동패턴 특성 분석 시작")
print("="*60)

BASE_PATH =r"E:\박정민\data\persona\pattern"
INPUT_FILE = os.path.join(BASE_PATH,"01_all_persona_combinations.csv")
OUTPUT_ALL = os.path.join(BASE_PATH,"07_profile_behavior_lift_all.csv")
OUTPUT_PROFILE = os.path.join(BASE_PATH,"08_profile_summary.csv")
OUTPUT_TOP = os.path.join(BASE_PATH,"09_profile_behavior.csv")

print("\n[06 전체 조합 결과 불러오기]")
df=pd.read_csv(INPUT_FILE,encoding="utf-8-sig")
print(f"행 수: {len(df):,}")
print(f"컬럼 수 : {len(df.columns)}")
print(df.columns.tolist())

EXPECTED_TOTAL = 4_405_673
total_count = df["건수"].sum()
print("\n"+"="*60)
print("무결성 확인")
print("="*60)
print(f"전체 건수 합계 : "
      f"{total_count:,}")
print(f"기대 모집단: "
      f"{EXPECTED_TOTAL:,}")
if total_count == EXPECTED_TOTAL:
    print("->전체 건수 일치")
else:
    print(f"->주의: "
          f"{total_count - EXPECTED_TOTAL:+,}건 차이")
    
profile_cols=["장애유형","휠체어"]
behavior_cols=["시간구간","요일구분","거리구간","공간이동"]
print("\n"+"="*60)
print("분석 구조")
print("="*60)
print("프로필 변수 : " + "x".join(profile_cols))
print("행동 변수 : " + "x".join(behavior_cols))

profile_total = (df.groupby(profile_cols,dropna=False)["건수"].sum().reset_index(name="프로필전체건수"))
profile_total["프로필전체비율(%)"]=(profile_total["프로필전체건수"]/total_count *100)
profile_total=profile_total.sort_values("프로필전체건수",ascending=False).reset_index(drop=True)
print("\n"+"="*60)
print("프로필 규모")
print("="*60)
print(profile_total.to_string(index=False))

behavior_total = (df.groupby(behavior_cols,dropna=False)["건수"].sum().reset_index(name="행동전체건수"))
behavior_total["행동전체비율(%)"]=(behavior_total["행동전체건수"]/total_count*100)

profile_behavior = (df.groupby(profile_cols + behavior_cols, dropna=False)["건수"].sum().reset_index(name="건수"))
profile_behavior = profile_behavior.merge(profile_total,on=profile_cols,how="left")
profile_behavior = profile_behavior.merge(behavior_total,on=behavior_cols,how="left")

profile_behavior["프로필내비율(%)"]=(profile_behavior["건수"]/profile_behavior["프로필전체건수"]*100)
profile_behavior["Lift"] = (profile_behavior["프로필내비율(%)"]/profile_behavior["행동전체비율(%)"])

profile_behavior["기대건수"] = profile_behavior["프로필전체건수"]*profile_behavior["행동전체비율(%)"]/100
profile_behavior["기대대비차이"]=(profile_behavior["건수"]-profile_behavior["기대건수"])

profile_behavior["프로필내빈도순위"] = (profile_behavior.groupby(profile_cols)["건수"].rank(method="first",ascending=False).astype(int))
profile_behavior["프로필내Lift순위"] = (profile_behavior.groupby(profile_cols)["Lift"].rank(method="first",ascending=False).astype(int))

profile_behavior = profile_behavior.sort_values(profile_cols + ["프로필내빈도순위"]).reset_index(drop=True)

profile_behavior.to_csv(OUTPUT_ALL,index=False,encoding="utf-8-sig")
profile_total.to_csv(OUTPUT_PROFILE,index=False,encoding="utf-8-sig")

top_frequency = (profile_behavior[profile_behavior["프로필내빈도순위"]<=5].copy())
top_frequency["선정기준"]="빈도Top5"
top_lift=(profile_behavior[profile_behavior["프로필내Lift순위"]<=5].copy())
top_lift["선정기준"]="LiftTop5"
top_result = pd.concat([top_frequency,top_lift],ignore_index=True)
top_result.to_csv(OUTPUT_TOP,index=False,encoding="utf-8-sig")

DISPLAY_PROFILE_SHARE = 0.5
major_profiles = profile_total[profile_total["프로필전체비율(%)"]>=DISPLAY_PROFILE_SHARE].copy()
print("\n"+"="*60)
print(f"주요 프로필"
      f"(전체 비율 {DISPLAY_PROFILE_SHARE}% 이상)")
print("="*60)
print(major_profiles.to_string(index=False))

print("\n"+"="*60)
print("주요 프로필별 많이 관찰된 행동 Top 5")
print("="*60)
for _,row in major_profiles.iterrows():
    disability = row["장애유형"]
    wheelchair = row["휠체어"]
    temp = profile_behavior[(profile_behavior["장애유형"]==disability)&(profile_behavior["휠체어"]==wheelchair)].copy()
    temp = temp.sort_values("건수",ascending = False).head(5)
    print(f"\n[{disability}/{wheelchair}]")
    print(temp[behavior_cols+["건수","프로필내비율(%)","행동전체비율(%)","Lift"]].round(3).to_string(index=False))

MIN_WITHIN_PROFILE_SHARE = 0.5
print("\n"+"="*60)
print(f"주요 프로필별 특징 행동"
      f"(프로필내 {MIN_WITHIN_PROFILE_SHARE}% 이상)")
print("="*60)
for _,row in major_profiles.iterrows():
    disability = row["장애유형"]
    wheelchair = row["휠체어"]
    temp = profile_behavior[(profile_behavior["장애유형"]==disability)&(profile_behavior["휠체어"]==wheelchair)&(profile_behavior["프로필내비율(%)"]>=MIN_WITHIN_PROFILE_SHARE)].copy()
    temp = temp.sort_values(["Lift","건수"],ascending = [False,False]).head(5)
    print(f"\n[{disability}/{wheelchair}]")
    print(temp[behavior_cols+["건수","프로필내비율(%)","행동전체비율(%)","Lift"]].round(3).to_string(index=False))

print("\n"+"="*60)
print("결과 저장")
print("="*60)
print(f"전체 프로필 x 행동 결과: \n"
      f"{OUTPUT_ALL}")
print(f"\n프로필 규모 요약: \n"
      f"{OUTPUT_PROFILE}")
print(f"프로필별 대표 행동: \n"
      f"{OUTPUT_TOP}")
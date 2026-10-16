import os
import pandas as pd

print("\n"+"="*60)
print("10 이동건 기반 페르소나 후보표 생성 시작")
print("="*60)
BASE_PATH =r"E:\박정민\data\persona\pattern"
PROFILE_FILE = os.path.join(BASE_PATH,"08_profile_summary.csv")
BEHAVIOR_FILE = os.path.join(BASE_PATH,"07_profile_behavior_lift_all.csv")
OUTPUT_FILE = os.path.join(BASE_PATH,"10_persona_candidate_summary.csv")
OUTPUT_DETAIL = os.path.join(BASE_PATH,"11_persona_candidate_top_behavior.csv")

print("\n[09 결과 불러오기]")
profile = pd.read_csv(PROFILE_FILE,encoding="utf-8-sig")
behavior = pd.read_csv(BEHAVIOR_FILE,encoding="utf-8-sig")
print(f"프로필 요약 행 수 : "
      f"{len(profile):,}")
print(f"프로필x행동 행 수 : "
      f"{len(behavior):,}")

MAJOR_SHARE = 0.5
major_profile = profile[profile["프로필전체비율(%)"]>=MAJOR_SHARE].copy()
major_profile=major_profile.sort_values("프로필전체비율(%)",ascending=False).reset_index(drop=True)
print("\n"+"="*60)
print("주요 프로필")
print("="*60)
print(major_profile.to_string(index=False))
major_share_sum = (major_profile["프로필전체비율(%)"].sum())
print(f"\n주요 프로필 수 : "
      f"{len(major_profile):,}개")
print(f"\n주요 프로필 누적 비율 : "
      f"{major_share_sum:.2f}%")

profile_cols = ["장애유형","휠체어"]
behavior_cols = ["시간구간","요일구분","거리구간","공간이동"]

total_n = behavior["건수"].sum()
overall_distribution = {}
for col in behavior_cols:
    temp = (behavior.groupby(col,dropna=False)["건수"].sum())
    temp = (temp/temp.sum())
    overall_distribution[col]=temp

def summarize_variable (temp,col):
    dist = (temp.groupby(col,dropna=False)["건수"].sum().reset_index())
    total=dist["건수"].sum()
    dist["프로필내비율(%)"]=(dist["건수"]/total*100)
    dist["전체비율(%)"]=(dist[col].map(overall_distribution[col])*100)
    dist["Lift"]=(dist["프로필내비율(%)"]/dist["전체비율(%)"])
    dominant = dist.sort_values("프로필내비율(%)",ascending = False).iloc[0]
    characteristic_pool = dist[dist["프로필내비율(%)"]>=1.0].copy()
    if len(characteristic_pool) == 0:
        characteristic_pool = dist.copy()
    characteristic = (characteristic_pool.sort_values(["Lift","프로필내비율(%)"],ascending=[False,False]).iloc[0])
    return {
        "대표값" : dominant[col],
        "대표비율" : dominant["프로필내비율(%)"],
        "특징값" : characteristic[col],
        "특징비율" : characteristic["프로필내비율(%)"],
        "특징Lift" : characteristic["Lift"]
    }

summary_rows=[]
detail_rows=[]
for _,p in major_profile.iterrows():
    disability = p["장애유형"]
    wheelchair = p["휠체어"]
    temp = behavior[(behavior["장애유형"]==disability)&(behavior["휠체어"]==wheelchair)].copy()
    freq_top = (temp.sort_values("건수",ascending=False).iloc[0])
    lift_pool = temp[temp["프로필내비율(%)"]>=0.5].copy()
    if len(lift_pool) == 0:
        lift_pool = temp.copy()
    lift_top = (lift_pool.sort_values(["Lift","건수"],ascending=[False,False]).iloc[0])
    time_info = summarize_variable(temp,"시간구간")
    day_info=summarize_variable(temp,"요일구분")
    distance_info = summarize_variable(temp,"거리구간")
    space_info = summarize_variable(temp,"공간이동")
    candidate_label = (
        f"{time_info['특징값']}"
        f"{distance_info['특징값']}"
        f"{space_info['특징값']}형"
    )
    summary_rows.append({
        "장애유형" : disability,
        "휠체어" : wheelchair,
        "프로필전체건수":p["프로필전체건수"],
        "프로필전체비율(%)":p["프로필전체비율(%)"],
        "대표시간":time_info["대표값"],
        "대표시간비율(%)":time_info["대표비율"],
        "대표요일":day_info["대표값"],
        "대표요일비율(%)":day_info["대표비율"],
        "대표거리":distance_info["대표값"],
        "대표거리비율(%)":distance_info["대표비율"],
        "대표공간":space_info["대표값"],
        "대표공간비율(%)":space_info["대표비율"],
        "특징시간":time_info["특징값"],
        "특징시간Lift":time_info["특징Lift"],
        "특징요일":day_info["특징값"],
        "특징요일Lift":day_info["특징Lift"],
        "특징거리":distance_info["특징값"],
        "특징거리Lift":distance_info["특징Lift"],
        "특징공간":space_info["특징값"],
        "특징공간Lift":space_info["특징Lift"],
        "대표행동":(
            f"{freq_top['시간구간']} / "
            f"{freq_top['요일구분']} / "
            f"{freq_top['거리구간']} / "
            f"{freq_top['공간이동']} / "
        ),
        "대표행동비율(%)":
        freq_top["프로필내비율(%)"],
        "대표행동Lift":
        freq_top["Lift"],
        "특징행동":
        (
            f"{lift_top['시간구간']} / "
            f"{lift_top['요일구분']} / "
            f"{lift_top['거리구간']} / "
            f"{lift_top['공간이동']} / "
        ),
        "특징행동비율(%)": lift_top["프로필내비율(%)"],
        "특징행동Lift":lift_top["Lift"],
        "행동아키타입후보":candidate_label
    })
    top3=(temp.sort_values("건수",ascending=False).head(3).copy())
    for rank, (_,r) in enumerate(top3.iterrows(),start=1):
        detail_rows.append({
            "장애유형":disability,
            "휠체어":wheelchair,
            "순위":rank,
            "시간구간":r["시간구간"],
            "요일구분":r["요일구분"],
            "거리구간":r["거리구간"],
            "공간이동":r["공간이동"],
            "건수":r["건수"],
            "프로필내비율(%)":r["프로필내비율(%)"],
            "Lift":r["Lift"]
        })

summary = pd.DataFrame(summary_rows)
summary = summary.sort_values("프로필전체비율(%)",ascending=False).reset_index(drop=True)
detail = pd.DataFrame(detail_rows)

round_cols = [col for col in summary.columns if( "비율" in col or "Lift" in col)]
for col in round_cols:
    summary[col] = (summary[col].astype(float).round(3))
detail["프로필내비율(%)"]=(detail["프로필내비율(%)"].round(3))
detail["Lift"]=(detail["Lift"].round(3))

print("\n"+"="*60)
print("페르소나 후보 요약")
print("="*60)

display_cols = ["장애유형","휠체어","프로필전체비율(%)","대표시간","대표거리","대표공간","특징시간","특징시간Lift","특징공간","특징공간Lift","행동아키타입후보"]
print(summary[display_cols].to_string(index=False))

print("\n"+"="*60)
print("프로필별 대표행동 / 특징행동")
print("="*60)
for _,row in summary.iterrows():
    print(f"\n[{row['장애유형']}] / "
          f"{row['휠체어']}")
    print(f"전체 비율 : "
          f"{row['프로필전체비율(%)']:.3f}%")
    print(f"대표 행동 : "
          f"{row['대표행동']}"
          f"{row['대표행동비율(%)']:.3f}%, "
          f"Lift{row['대표행동Lift']:.3f}")
    print(f"특징 행동 : "
          f"{row['특징행동']}"
          f"{row['특징행동비율(%)']:.3f}%, "
          f"Lift {row['특징행동Lift']:.3f}")
    print(f"아키타입 후보 : "
          f"{row['행동아키타입후보']}")

print("\n"+"="*60)
print("행동 아키타입 후보별 프로필")
print("="*60)
group_summary = (summary.groupby("행동아키타입후보",dropna=False).agg(프로필수=("장애유형","size"),누적비율 = ("프로필전체비율(%)","sum")).reset_index())
group_summary = group_summary.sort_values("누적비율",ascending=False)
for _,g in group_summary.iterrows():
    label = g["행동아키타입후보"]
    temp = summary[summary["행동아키타입후보"]==label]
    profile_text = ", ".join((temp["장애유형"]+"."+temp["휠체어"]).tolist())
    print(f"\n[{label}]")
    print(f"프로필 : "
          f"{profile_text}")
    print(f"프로필 수 : "
          f"{int(g['프로필수'])}")
    print(f"누적 비율 : "
          f"{g['누적비율']:.2f}%")

summary.to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig")
detail.to_csv(OUTPUT_DETAIL,index=False,encoding="utf-8-sig")
print("\n"+"="*60)
print("결과 저장")
print("="*60)
print(f"페르소나 후보 요약:\n"
      f"{OUTPUT_FILE}")
print(f"\n프로필별 대표 행동 Top3:\n"
      f"{OUTPUT_DETAIL}")
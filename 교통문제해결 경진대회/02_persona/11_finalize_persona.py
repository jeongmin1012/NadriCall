import os
import pandas as pd

print("\n"+"="*60)
print("11 이동건 기반 최종 페르소나 정리 시작")
print("="*60)
BASE_PATH =r"E:\박정민\data\persona\pattern"
INPUT_FILE = os.path.join(BASE_PATH,"10_persona_candidate_summary.csv")
PROFILE_FILE = os.path.join(BASE_PATH,"08_profile_summary.csv")
OUTPUT_VARIANT = os.path.join(BASE_PATH,"12_final_persona_variants.csv")
OUTPUT_ARCHETYPE = os.path.join(BASE_PATH,"13_final_archetype_summary.csv")
OUTPUT_CARD = os.path.join(BASE_PATH,"14_persona_cards.txt")
OUTPUT_RARE = os.path.join(BASE_PATH,"15_rare_profiles.csv")

df = pd.read_csv(INPUT_FILE,encoding="utf-8-sig")
profile_all = pd.read_csv(PROFILE_FILE,encoding="utf-8-sig")
df["장애유형"]=(df["장애유형"].astype(str).str.strip())
df["휠체어"] = (df["휠체어"].astype(str).str.strip())
print(f"\n주요 프로필 수 : {len(df):,}개")
print(f"주요 프로필 누적 비율 : "
      f"{df['프로필전체비율(%)'].sum():.2f}%")

archetype_meta = {
    "A1":{
        "아키타입명": "노약자 중심 근거리 생활권형",
    "핵심설명": "노약자 프로필을 중심으로 짧은 거리와 동일 구군 이동이 대표적으로 관찰되며, 오전 이동이 상대적으로 특정적으로 나타나는 유형"
    },
    "A2":{
        "아키타입명":"신장 프로필 심야&새벽 특징형",
        "핵심설명":"신장 프로필의 대표 이동은 낮 시간에도 존재하지만, 전체 이동과 비교했을 때 심야&새벽 이동이 상대적으로 강하게 나타나는 유형"
    },
    "A3":{
        "아키타입명":"오후 근거리 생활권 비휠체어형",
        "핵심설명": "비휠체어 프로필을 중심으로 오후&단거리&동일 구군 이동이 대표적으로 관찰되는 생활권 이동 유형"
    },
    "A4":{
        "아키타입명":"오후 광역 중장거리 휠체어형",
        "핵심설명": "지체&뇌병변 휠체어 프로필을 중심으로 오후 시간대 중장거리 및 타 구군 이동이 대표적으로 관찰되는 유형"
    },
    "A5":{
        "아키타입명":"오후 광역 중장거리 비휠체어형",
        "핵심설명": "시각&지적&자폐성 비휠체어 프로필을 중심으로 오후 시간대 중장거리 및 타 구군 이동이 대표적으로 관찰되는 유형"
    }
}

profile_to_archetype={
    ("노약자","비휠체어"):"A1",
    ("노약자","휠체어"):"A1",
    ("신장","비휠체어"):"A2",
    ("신장","휠체어"):"A2",
    ("지체","비휠체어"):"A3",
    ("뇌병변","비휠체어"):"A3",
    ("정신","비휠체어"):"A3",
    ("청각","비휠체어"):"A3",
    ("지체","휠체어"):"A4",
    ("뇌병변","휠체어"):"A4",
    ("시각","비휠체어"):"A5",
    ("지적","비휠체어"):"A5",
    ("자폐성","비휠체어"):"A5",
}

def get_archetype(row):
    key = (row["장애유형"],row["휠체어"])
    return profile_to_archetype.get(key,"미분류")
df["아키타입ID"] = df.apply(get_archetype,axis=1)
df["아키타입명"]=df["아키타입ID"].apply(lambda x : (archetype_meta[x]["아키타입명"] if x in archetype_meta else "미분류"))


df = df.sort_values(["아키타입ID","프로필전체비율(%)"],ascending=[True,False]).reset_index(drop=True)
df["Persona_ID"] = [f"p{i:02d}" for i in range(1,len(df)+1)]

variant_cols=["Persona_ID","아키타입ID","아키타입명","장애유형","휠체어","프로필전체건수","프로필전체비율(%)","대표행동","대표행동비율(%)","대표행동Lift","특징행동","특징행동비율(%)","특징행동Lift","대표시간","대표요일","대표거리","대표공간"]
variant= df[variant_cols].copy()
variant.to_csv(OUTPUT_VARIANT,index=False,encoding="utf-8-sig")

archetype_rows=[]
for archetype_id,meta in archetype_meta.items():
    temp = df[df["아키타입ID"]== archetype_id].copy()
    profiles = ", ".join((temp["장애유형"]+"."+temp["휠체어"]).tolist())
    archetype_rows.append({
        "아키타입ID":archetype_id,
        "아키타입명":meta["아키타입명"],
        "포함프로필":profiles,
        "프로필수":len(temp),
        "포함프로필건수":temp["프로필전체건수"].sum(),
        "포함프로필비율(%)":temp["프로필전체비율(%)"].sum(),
        "핵심설명":meta["핵심설명"]
    })
archetype = pd.DataFrame(archetype_rows)
archetype["포함프로필비율(%)"] = (archetype["포함프로필비율(%)"].round(3))
archetype.to_csv(OUTPUT_ARCHETYPE,index=False,encoding="utf-8-sig")

major_keys = set(zip(df["장애유형"],df["휠체어"]))
rare_mask = profile_all.apply(lambda row:(row["장애유형"],row["휠체어"])not in major_keys,axis=1)
rare = profile_all[rare_mask].copy()
rare=rare.sort_values("프로필전체비율(%)",ascending=False)
rare.to_csv(OUTPUT_RARE,index=False,encoding="utf-8-sig")

print("\n"+"="*60)
print("최종 행동 아키타입")
print("="*60)

print(archetype.to_string(index=False))
print("\n"+"="*60)
print("최종 세부 페르소나")
print("="*60)
print(variant[["Persona_ID","아키타입ID","장애유형","휠체어","프로필전체비율(%)","대표행동","특징행동"]].to_string(index=False))
cards=[]
for _,row in df.iterrows():
    archetype_id = row["아키타입ID"]
    meta = archetype_meta[archetype_id]
    card = f"""
=====================================================
{row['Persona_ID']}|{meta['아키타입명']}
=====================================================
[프로필]
장애유형: {row['장애유형']}
휠체어: {row['휠체어']}

[데이터 내 비중]
전체 완료 이동건 중:
{row['프로필전체비율(%)']:.3f}%

[대표행동]
{row['대표행동']}
프로필 내 비율:
{row['대표행동비율(%)']:.3f}%

[상대적으로 특징적인 행동]
{row['특징행동']}
프로필 내 비율:
{row['특징행동비율(%)']:.3f}%
전제 대비 Lift:
{row['특징행동Lift']:.3f}

[아키타입 해석]
{meta['핵심설명']}

[해석주의]
이 페르소나는 실제 특정 개인을 의미하지 않는다.
개인식별자가 없는 나드리콜 완료 이동건에서
반복적으로 관찰된 프로필 및 행동특성을
대표적으로 정리한 이동건 기반 페르소나이다.
"""
    
    cards.append(card.strip())
with open(OUTPUT_CARD,"w",encoding="utf-8") as f:
    f.write("\n\n".join(cards))

print("\n"+"="*60)
print("결과 저장")
print("="*60)

print(f"최종 세부 페르소나: \n"
      f"{OUTPUT_VARIANT}")
print(f"\n최종 행동 아키타입:\n"
      f"{OUTPUT_ARCHETYPE}")
print(f"\n페르소나  카드:\n"
      f"{OUTPUT_CARD}")
print(f"\n희귀 프로필 보존:\n"
      f"{OUTPUT_RARE}")

print("\n"+"="*60)
print("11 이동건 기반 최종 페르소나 정리 완료")
print("="*60)

print("""
현재 결과
- 핵심 행동 아키타입 5개
- 주요 장애유형 x 휠체어 세부 페르소나 13개
- 주요 프로필은 전체 완료 이동건의 약 97.45%를 포함
- 나머지 희귀 프로필은 삭제하지 않고 별도 보존

페르소나의 의미
- 실제 이용자를 복원한 결과가 아님
- 개인별 반복 이동을 의미하지 않음
- 완료 이동건에서 반복적으로 나타나는 프로필과 행동특성을 대표적으로 정리한 이동건 기반 데이터 페르소나

다음 작업
- 생성된 페르소나 카드 검토
- 보고서용 표현 정리
- LLM 가상설무용 프롬포트 구조 설계""")
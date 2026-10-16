import os
import numpy as np
import pandas as pd

BASE_PATH =r"E:\박정민\data\persona\pattern"
BEHAVIOR_FILE = os.path.join(BASE_PATH,"07_profile_behavior_lift_all.csv")
PERSONA_FILE = os.path.join(BASE_PATH,"12_final_persona_variants.csv")
OUTPUT_FILE = os.path.join(BASE_PATH,"16_synthetic_population_10000.csv")
N_SYNTHETIC = 10_000
RANDOM_SEED = 42
EXPECTED_TOTAL = 4_405_673
print("\n"+"="*60)
print("12 합성 응답 에이전트 생성 시작")
print("="*60)

def read_csv_auto(path):
    encodings=["utf-8-sig","uft-8","cp949"]
    for enc in encodings:
        try:
            df = pd.read_csv(path,encoding=enc)
            print(f"[불러오기 성공]{os.path.basename(path)}/encoding={enc}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"파일 인코딩을 확인하지 못했습니다: {path}")
behavior = read_csv_auto(BEHAVIOR_FILE)
persona = read_csv_auto(PERSONA_FILE)
print("\n[07 데이터]")
print("행 수:",len(behavior))
print("컬럼:",behavior.columns.tolist())
print("\n[최종 페르소나]")
print("행 수:",len(persona))
print("컬럼:",persona.columns.tolist())

behavior_required = ["장애유형","휠체어","시간구간","요일구분","거리구간","공간이동","건수"]
persona_required = ["Persona_ID","아키타입ID","아키타입명","장애유형","휠체어"]
missing_behavior = [col for col in behavior_required if col not in behavior.columns]
missing_persona = [col for col in persona_required if col not in persona.columns]
if missing_behavior:
    raise KeyError(f"07 파일에 필요한 컬럼 없음: {missing_behavior}")
if missing_persona:
    raise KeyError(f"12 파일에 필요한 컬럼 없음: {missing_persona}")

text_cols=["장애유형","휠체어","시간구간","요일구분","거리구간","공간이동"]
for col in text_cols:
    if col in behavior.columns:
        behavior[col] = (behavior[col].astype(str).str.strip())
for col in ["장애유형","휠체어"]:
    persona[col] = (persona[col].astype(str).str.strip())
behavior = behavior[behavior["건수"]>0].copy()

persona_map = (persona [["Persona_ID","아키타입ID","아키타입명","장애유형","휠체어"]].drop_duplicates())
dup = persona_map.duplicated(subset=["장애유형","휠체어"],keep=False)
if dup.any():
    print("\n[중복 프로필 발견]")
    print(persona_map.loc[dup,["Persona_ID","장애유형","휠체어"]])
    raise ValueError("장애유형x휠체어 프로필이 여러 Persona_ID에 연결되어 있습니다.")
df = behavior.merge(persona_map,on=["장애유형","휠체어"],how="left",validate="many_to_one")

df["프로필구분"] = np.where(df["Persona_ID"].notna(),"주요13","희귀")
df["Persona_ID"] = (df["Persona_ID"].fillna("RARE"))
df["아키타입ID"] = (df["아키타입ID"].fillna("RARE"))
df["아키타입명"] = (df["아키타입명"].fillna("희귀 프로필"))

totla_count = int(df["건수"].sum())
print("\n"+"="*60)
print("원본 집계 확인")
print("="*60)
print(f"07 전체 건수 합계:{totla_count:,}")
print(f"기대 Persona Base: {EXPECTED_TOTAL:,}")
if totla_count == EXPECTED_TOTAL:
    print("->전체 건수 일치")
else:
    print(f"->주의: {totla_count - EXPECTED_TOTAL:+,}건 차이")
profile_type = (df.groupby("프로필구분")["건수"].sum().reset_index())
profile_type["비율(%)"] = (profile_type["건수"]/totla_count*100)
print("\n[주요 / 희귀 프로필 비율]")
print(profile_type.to_string(index=False))

weights = (df["건수"].to_numpy(dtype=float))
probabilities = (weights/weights.sum())

rng = np.random.default_rng(RANDOM_SEED)
generated_counts = rng.multinomial(N_SYNTHETIC,probabilities)
row_index = np.repeat(np.arange(len(df)),generated_counts)
synthetic = (df.iloc[row_index].copy().reset_index(drop=True))
random_order = rng.permutation(len(synthetic))
synthetic = (synthetic.iloc[random_order].reset_index(drop=True))

synthetic.insert(0,"Synthetic_ID",[f"S{i:05d}" for i in range(1,len(synthetic)+1)])

final_cols = ["Synthetic_ID","Persona_ID","아키타입ID","아키타입명","프로필구분","장애유형","휠체어","시간구간","요일구분","거리구간","공간이동"]
synthetic = synthetic[final_cols].copy()

print("\n"+"="*60)
print("합성인구 생성 결과")
print("="*60)
print(f"합성 에이전트 수:{len(synthetic):,}")
print("\n[주요/희귀]")
synthetic_type = (synthetic["프로필구분"].value_counts().rename_axis("프로필구분").reset_index(name = "합성건수"))
synthetic_type["합성비율(%)"] = (synthetic_type["합성건수"]/len(synthetic)*100)
print(synthetic_type.to_string(index=False))
print("\n[Persona별 생성 수]")
persona_check = (synthetic.groupby(["Persona_ID","아키타입ID","장애유형","휠체어"],dropna=False).size().reset_index(name="합성건수"))
persona_check["합성비율(%)"] = (persona_check["합성건수"]/len(synthetic)*100)
print(persona_check.to_string(index=False))
print("\n[샘플 20개]")
print(synthetic.head(20).to_string(index=False))

synthetic.to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig")
print("\n"+"="*60)
print("저장 완료")
print("="*60)
print(OUTPUT_FILE)
print("\n12 합성 응답 에이전트 생성 완료")
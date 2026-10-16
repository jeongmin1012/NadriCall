import os
import numpy as np
import pandas as pd

BASE_PATH =r"E:\박정민\data\persona\pattern"
ORIGINAL_FILE = os.path.join(BASE_PATH,"07_profile_behavior_lift_all.csv")
PERSONA_FILE = os.path.join(BASE_PATH,"12_final_persona_variants.csv")
SYNTHETIC_FILE = os.path.join(BASE_PATH,"16_synthetic_population_10000.csv")
OUTPUT_FILE = os.path.join(BASE_PATH,"17_synthetic_profile_audit.csv")
OUTPUT_BEHAVIOR = os.path.join(BASE_PATH,"18_synthetic_behavior_audit.csv")
OUTPUT_JOINT = os.path.join(BASE_PATH,"19_synthetic_joint_audit.csv")
OUTPUT_SUMMARY = os.path.join(BASE_PATH,"20_synthetic_audit_summary.csv")
EXPECTED_TOTAL = 4_405_673

print("\n"+"="*60)
print("13 합성인구 Audit 시작")
print("="*60)

def read_csv_auto(path):
    encodings = ["utf-8-sig","utf-8","cp949"]
    for enc in encodings:
        try:
            df = pd.read_csv(path,encoding=enc)
            print(f"[불러오기 성공]"
                  f"{os.path.basename(path)}/encoding={enc}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"파일 인코딩 확인 실패: {path}")
original = read_csv_auto(ORIGINAL_FILE)
persona= read_csv_auto(PERSONA_FILE)
synthetic = read_csv_auto(SYNTHETIC_FILE)
print("\n[원본 집계]")
print("행 수:",len(original))
print("\n[최종 페르소나]")
print("행 수:",len(persona))
print("\n[합성인구]")
print("행 수:",len(synthetic))

profile_cols = ["장애유형","휠체어"]
behavior_cols = ["시간구간","요일구분","거리구간","공간이동"]
joint_cols = (profile_cols + behavior_cols)

for col in joint_cols:
    original[col] = (original[col].astype(str).str.strip())
    synthetic[col] = (synthetic[col].astype(str).str.strip())
persona["장애유형"] = (persona["장애유형"].astype(str).str.strip())
persona["휠체어"] = (persona["휠체어"].astype(str).str.strip())
original["건수"] = pd.to_numeric(original["건수"],errors="coerce").fillna(0)
original = original[original["건수"]>0].copy()

original_total = int(original["건수"].sum())
synthetic_total = len(synthetic)
print("\n"+"="*60)
print("기본 건수 확인")
print("="*60)
print(f"원본 Persona Base: {original_total:,}")
print(f"기대 Persona Base: {EXPECTED_TOTAL:,}")
print(f"합성 에이전트 : {synthetic_total:,}")
if original_total == EXPECTED_TOTAL:
    print("->원본 건수 일치")
else:
    print(f"->주의: {original_total - EXPECTED_TOTAL:+,}건 차이")

persona_map = (persona[["Persona_ID","아키타입ID","아키타입명","장애유형","휠체어"]].drop_duplicates(subset=["장애유형","휠체어"]))
original = original.merge(persona_map,on=["장애유형","휠체어"],how="left",validate="many_to_one")
original["Persona_ID"] = (original["Persona_ID"].fillna("RARE"))
original["아키타입ID"] = (original["아키타입ID"].fillna("RARE"))
original["아키타입명"] = (original["아키타입명"].fillna("희귀 프로필"))

original["프로필구분"] = np.where(original["Persona_ID"]=="RARE","희귀","주요13")
if "프로필구분" not in synthetic.columns:
    synthetic["프로필구분"] = np.where(synthetic["Persona_ID"]=="RARE","희귀","주요13")

print("\n"+"="*60)
print("1. 프로필 분포 Audit")
print("="*60)
original_profile = (original.groupby(["Persona_ID","장애유형","휠체어"],dropna=False)["건수"].sum().reset_index(name="원본건수"))
original_profile["원본비율(%)"] = (original_profile["원본건수"]/original_total*100)
synthetic_profile = (synthetic.groupby(["Persona_ID","장애유형","휠체어"],dropna=False).size().reset_index(name="합성건수"))
synthetic_profile["합성비율(%)"] = (synthetic_profile["합성건수"]/synthetic_total*100)
profile_audit = original_profile.merge(synthetic_profile,on=["Persona_ID","장애유형","휠체어"],how="outer")
for col in ["원본건수","원본비율(%)","합성건수","합성비율(%)"]:
    profile_audit[col] =(profile_audit[col].fillna(0))
profile_audit["차이(%p)"] = (profile_audit["합성비율(%)"]-profile_audit["원본비율(%)"])
profile_audit["절대차이(%p)"] = (profile_audit["차이(%p)"].abs())
profile_audit = profile_audit.sort_values("원본비율(%)",ascending=False).reset_index(drop=True)
print(profile_audit[["Persona_ID","장애유형","휠체어","원본비율(%)","합성비율(%)","차이(%p)"]].head(30).to_string(index=False))
profile_mae = (profile_audit["절대차이(%p)"].mean())
profile_max = (profile_audit["절대차이(%p)"].max())
print(f"\n프로필 평균 절대 차이(MAE): {profile_mae:.4f}%p")
print(f"프로필 최대 절대 차이: {profile_max:.4f}%p")

original_type = (original.groupby("프로필구분")["건수"].sum().reset_index(name="원본건수"))
original_type["원본비율(%)"] = (original_type["원본건수"]/original_total*100)
synthetic_type = (synthetic.groupby("프로필구분").size().reset_index(name="합성건수"))
synthetic_type["합성비율(%)"] = (synthetic_type["합성건수"]/synthetic_total*100)
type_audit = original_type.merge(synthetic_type,on="프로필구분",how="outer")
type_audit["차이(%p)"] = (type_audit["합성비율(%)"]-type_audit["원본비율(%)"])
print("\n[주요13/희귀]")
print(type_audit.to_string(index=False))

print("\n"+"="*60)
print("2. 행동 변수 분포 Auidt")
print("="*60)
behavior_results=[]
behavior_summary=[]
for variable in behavior_cols:
    original_dist = (original.groupby(variable)["건수"].sum().reset_index(name="원본건수"))
    original_dist["원본비율(%)"] = (original_dist["원본건수"]/original_total*100)
    synthetic_dist = (synthetic.groupby(variable).size().reset_index(name="합성건수"))
    synthetic_dist["합성비율(%)"] = (synthetic_dist["합성건수"]/synthetic_total*100)
    temp = original_dist.merge(synthetic_dist,on=variable,how="outer")
    for col in ["원본건수","원본비율(%)","합성건수","합성비율(%)"]:
        temp[col] = (temp[col].fillna(0))
    temp["차이(%p)"] = (temp["합성비율(%)"]-temp["원본비율(%)"])
    temp["절대차이(%p)"] = (temp["차이(%p)"].abs())
    temp.insert(0,"변수",variable)
    temp = temp.rename(columns={variable:"범주"})
    behavior_results.append(temp)
    mae = (temp["절대차이(%p)"].mean())
    max_diff = (temp["절대차이(%p)"].max())
    behavior_summary.append({
        "구분":variable,
        "MAE(%p)":mae,
        "최대절대차이(%p)":max_diff
    })
    print(f"\n[{variable}]")
    print(temp[["범주","원본비율(%)","합성비율(%)","차이(%p)"]].to_string(index=False))
    print(f"MAE: {mae:.4f}%p")
behavior_audit = pd.concat(behavior_results,ignore_index=True)

print("\n"+"="*60)
print("3. 6변수 결합분포 Audit")
print("="*60)
original_joint= (original.groupby(joint_cols,dropna=False)["건수"].sum().reset_index(name="원본건수"))
original_joint["원본비율(%)"] = (original_joint["원본건수"]/original_total*100)
synthetic_joint = (synthetic.groupby(joint_cols,dropna=False).size().reset_index(name="합성건수"))
synthetic_joint["합성비율(%)"] = (synthetic_joint["합성건수"]/synthetic_total*100)
joint_audit = original_joint.merge(synthetic_joint,on=joint_cols,how="outer")
for col in ["원본건수","원본비율(%)","합성건수","합성비율(%)"]:
    joint_audit[col]=(joint_audit[col].fillna(0))
joint_audit["차이(%p)"] = (joint_audit["합성비율(%)"]-joint_audit["원본비율(%)"])
joint_audit["절대차이(%p)"] = (joint_audit["차이(%p)"].abs())
joint_audit = joint_audit.sort_values("원본비율(%)",ascending=False).reset_index(drop=True)
joint_mae = (joint_audit["절대차이(%p)"].mean())
joint_max = (joint_audit["절대차이(%p)"].max())
not_sampled = ((joint_audit["원본건수"]>0)&(joint_audit["합성건수"]==0)).sum()
print(f"원본 결합조합 수 : {len(original_joint):,}")
print(f"합성 결합조합 수 : {len(synthetic_joint):,}")
print(f"합성표본에 미출현한 원본 조합 : {not_sampled:,}")
print(f"\n결합분포 평균 절대 차이(MAE): {joint_mae:.6f}")
print(f"결합분포 최대 절대 차이: {joint_max:.4f}%p")
print("\n[원본 비중 상위 20개 결합조합]")
print(joint_audit[joint_cols+["원본비율(%)","합성비율(%)","차이(%p)"]].head(20).to_string(index=False))

summary_rows = [{
    "구분" : "프로필",
    "MAE(%p)" : profile_mae,
    "최대절대차이(%p)" : profile_max
}
]
summary_rows.extend(behavior_summary)
summary_rows.append({
    " 구분":"6변수 결합분포",
    "MAE(%p)" : joint_mae,
    "최대절대차이(%p)":joint_max
})
audit_summary = pd.DataFrame(summary_rows)
print("\n"+"="*60)
print("Audit 요약")
print("="*60)
print(audit_summary.to_string(index=False))

profile_audit.to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig")
behavior_audit.to_csv(OUTPUT_BEHAVIOR,index=False,encoding="utf-8-sig")
joint_audit.to_csv(OUTPUT_JOINT,index=False,encoding="utf-8-sig")
audit_summary.to_csv(OUTPUT_SUMMARY,index=False,encoding="utf-8-sig")

print("\n"+"="*60)
print("저장 완료")
print("="*60)
print("17 프로필 Audit: ",OUTPUT_FILE)
print("18 행동 Audit: ",OUTPUT_BEHAVIOR)
print("19 결합분포 Audit: ",OUTPUT_JOINT)
print("20 Audit 요약:",OUTPUT_SUMMARY)
print("\n13 합성인구 Audit 완료")
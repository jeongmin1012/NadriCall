import os
import numpy as np
import pandas as pd

DATA_PATH = r"E:\박정민\data"
INPUT_FILE = os.path.join(DATA_PATH,"clean","nadricall_2022_2025_clean.csv")
SAVE_PATH = os.path.join(DATA_PATH,"persona")
os.makedirs(SAVE_PATH,exist_ok=True)

PERSONA_FILE = os.path.join(SAVE_PATH,"persona_base_2022_2024.csv")
DISTRIBUTION_FILE = os.path.join(SAVE_PATH,"persona_variable_distribution.csv")

print("\n"+"="*60)
print("05 페르소나 분석 시작")
print("="*60)
print("\n[clean] 데이터 불러오기")
usecols=["연도","상태","완료이동여부","장애유형","휠체어 유무","접수시간대","접수요일","접수평일주말","이동거리","호출_지역구분","호출_구군","목적지_지역구분","목적지_구군"]
df=pd.read_csv(INPUT_FILE,usecols=usecols,encoding="utf-8-sig",low_memory=False)
print(f"전체 clean 데이터: "
      f"{len(df):,}건")
print(f"불러온 컬럼 수: "
      f"{len(df.columns)}개")

print("\n"+"="*60)
print("기본 자료형 정리")
print("="*60)
numeric_cols=["연도","완료이동여부","접수시간대","이동거리"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col],errors="coerce")

string_cols=["상태","장애유형","휠체어 유무","접수요일","접수평일주말","호출_지역구분","호출_구군","목적지_지역구분","목적지_구군"]
for col in string_cols:
    df[col]=(df[col].astype("string").str.strip())

print("\n"+"="*60)
print("페르소나 기본 모집단 추출")
print("="*60)

base = df[(df["연도"].isin([2022,2023,2024]))&(df["완료이동여부"]==1)].copy()
print(f"\n기본 모집단: "
      f"{len(base):,}건")
print("\n[연도별 완료 이동]")
print(base["연도"].value_counts().sort_index())
print("\n[상태 확인]")
print(base["상태"].value_counts(dropna=False))

print("\n"+"="*60)
print("핵심 변수 데이터 품질 확인")
print("="*60)
check_cosl=["장애유형","휠체어 유무","접수시간대","접수요일","접수평일주말","이동거리","호출_지역구분","호출_구군","목적지_지역구분","목적지_구군"]
quality=pd.DataFrame({"결측건수":base[check_cosl].isna().sum(),"결측률(%)":(base[check_cosl].isna().mean()*100).round(4)})
print(quality)

print("\n"+"="*60)
print("시간구간 생성")
print("="*60)
time_bins=[-1,5,8,11,16,19,23]
time_labels=["심야&새벽","이른 오전","오전","오후","저녁","야간"]
base["시간구간"]=pd.cut(base["접수시간대"],bins=time_bins,labels=time_labels,include_lowest=True)
print("\n[시간구간 건수]")
print(base["시간구간"].value_counts(sort=False,dropna=False))
print("\n[시간구간 비율(%)]")
print((base["시간구간"].value_counts(normalize=True,sort=False,dropna=False)*100).round(2))

print("\n"+"="*60)
print("요일구분 생성")
print("="*60)
base["요일구분"]=(base["접수평일주말"])
print("\n[요일구분 건수]")
print(base["요일구분"].value_counts(dropna=False))
print("\n[요일구분 비율(%)]")
print((base["요일구분"].value_counts(normalize=True,dropna=False)*100).round(2))

print("\n"+"="*60)
print("이동거리 확인")
print("="*60)
print(f"\n이동거리 결측: "
      f"{base["이동거리"].isna().sum():,}건")
print(f"이동거리 0m: "
      f"{(base["이동거리"]==0).sum():,}건")
print(f"이동거리 음수: "
      f"{(base["이동거리"]<0).sum():,}건")

print("\n"+"="*60)
print("페르소나용 거리 기준 계산")
print("="*60)
distance_valid=(base["이동거리"].dropna())
distance_valid=distance_valid[distance_valid>0]
q25=distance_valid.quantile(0.25)
q50=distance_valid.quantile(0.50)
q75=distance_valid.quantile(0.75)
q90=distance_valid.quantile(0.90)
print(f"\n25%: "
      f"{q25:,.0f}m")
print(f"\n50%: "
      f"{q50:,.0f}m")
print(f"\n75%: "
      f"{q75:,.0f}m")
print(f"\n90%: "
      f"{q90:,.0f}m")

distance_bins=[0,q25,q50,q75,q90,np.inf]
distance_labels=["0~25%","25~50%","50~75%","75~90%","상위10%"]
base["거리구간"]=pd.cut(base["이동거리"],bins=distance_bins,labels=distance_labels,include_lowest=False)
print("\n[거리구간 건수]")
print(base["거리구간"].value_counts(sort=False,dropna=False))
print("\n[거리구간 비율(%)]")
print((base["거리구간"].value_counts(normalize=True,sort=False,dropna=False)*100).round(2))

print("\n"+"="*60)
print("공간이동 유형 생성")
print("="*60)
def classify_spatial(row):
    origin_region=row["호출_지역구분"]
    dest_region=row["목적지_지역구분"]
    origin_gu=row["호출_구군"]
    dest_gu=row["목적지_구군"]
    if(pd.isna(origin_region) or pd.isna(dest_region)):
        return pd.NA
    invalid_region=["미상","주소코드"]
    if(origin_region in invalid_region or dest_region in invalid_region):
        return pd.NA
    if(origin_region == "대구" and dest_region == "대구"):
        if(pd.isna(origin_gu) or pd.isna(dest_gu)):
            return pd.NA
        if origin_gu == dest_gu:
            return "동일 구군"
        else:
            return "대구 내 타 구군"
        return "대구 외 이동"
base["공간이동"] = base.apply(classify_spatial, axis=1)
print("\n[공간이동 건수]")
print(base["공간이동"].value_counts(dropna=False))
print("\n[공간이동 비율(%)]")
print((base["공간이동"].value_counts(normalize=True,dropna=False)*100).round(2))

print("\n"+"="*60)
print("휠체어 변수 확인")
print("="*60)
base["휠체어"]=(base["휠체어 유무"])
print(base["휠체어"].value_counts(dropna=False))

print("\n"+"="*60)
print("휠체어 변수 확인")
print("="*60)
print(base["장애유형"].value_counts(dropna=False))

print("\n"+"="*60)
print("최종 분석 가능 데이터 추출")
print("="*60)
persona_cols=["장애유형","휠체어","시간구간","요일구분","거리구간","공간이동"]
valid_mask=(base[persona_cols].notna().all(axis=1))
persona=base.loc[valid_mask].copy()
print(f"\n기본 모집단: "
      f"{len(base):,}건")
print(f"최종 분석 가능: "
      f"{len(persona):,}건")
print(f"제외: "
      f"{len(base)-len(persona):,}건")
print(f"유지율: "
      f"{len(persona)/len(base)*100:.2f}%")

print("\n"+"="*60)
print("Persona Base Table 생성")
print("="*60)
persona_base = persona[["연도","접수시간대","접수요일","접수평일주말","이동거리","호출_지역구분","호출_구군","목적지_지역구분","목적지_구군","장애유형","휠체어","시간구간","요일구분","거리구간","공간이동"]].copy()
print(f"\nPersona Base: "
      f"{len(persona_base):,}건")
print("\n[앞 10행]")
print(persona_base.head(10).to_string(index=False))

print("\n"+"="*60)
print("최종 결측 확인")
print("="*60)
missing_result = pd.DataFrame({"결측건수":persona_base[persona_cols].isna().sum()})
missing_result["결측률(%)"]=(missing_result["결측건수"]/len(persona_base)*100).round(4)
print(missing_result)

print("\n"+"="*60)
print("변수별 분포표 생성")
print("="*60)
distribution_list=[]
for col in persona_cols:
    temp = (persona_base[col].value_counts(dropna=False).rename_axis("값").reset_index(name="건수"))
    temp["비율(%)"]=(temp["건수"]/len(persona_base)*100).round(2)
    temp.insert(0,"변수",col)
    distribution_list.append(temp)
distribution_df=pd.concat(distribution_list,ignore_index=True)
print(distribution_df.to_string(index=False))

print("\n"+"="*60)
print("연도별 핵심 변수 기초 확인")
print("="*60)

for col in ["장애유형","휠체어","시간구간","요일구분","거리구간","공간이동"]:
    print(f"\n[연도 X {col}]")
    table=pd.crosstab(persona_base["연도"],persona_base[col],normalize="index")
    print(table.round(2))

print("\n"+"="*60)
print("연도별 핵심 변수 기초 확인")
print("="*60)
persona_base.to_csv(PERSONA_FILE,index=False,encoding="utf-8-sig")
distribution_df.to_csv(DISTRIBUTION_FILE,index=False,encoding="utf-8-sig")
print("\nPersona Base 저장: ")
print(PERSONA_FILE)
print("\n변수별 분포 저장:")
print(DISTRIBUTION_FILE)

print("\n"+"="*60)
print("Persona Base 생성완료")
print("="*60)
print(
    f"""분석단위
      - 완료 이동 1건
      분석기간
      - 2022~2024
      기본모집단
      - {len(base):,}건
      최종 Persona Base
      - {len(persona_base):,}건
      유지율
      - {len(persona_base)/len(base)*100:.2f}%
         핵심 후보 변수
         1. 장애유형
         2.휠체어
         3. 시간구간
         4. 요일구분
         5.거리구간
         6.공간이동
         주의
         - 이용자 ID가 없으므로 개인 단위 페르소나 군집이 아님
         - 이동 1건에서 함께 관찰되는 특성 조합을 분석함
         - 희귀 장애유형은 아직 임의 통합하지 않음
         - 취소는 완료 이동 페르소나의 핵심 변수에서 제외
         - 아직 군집화하지 않음
         - 아직 최종 페르소나를 확정하지 않음
         """)
print("="*60)
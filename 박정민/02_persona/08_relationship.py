import os
import numpy as np
import pandas as pd

print("\n"+"="*60)
print("08 페르소나 핵심 변수 관계 분석 시작")
print("="*60)

BASE_PATH =r"E:\박정민\data\persona\pattern"
INPUT_FILE = os.path.join(BASE_PATH,"01_all_persona_combinations.csv")
OUTPUT_MATRIX = os.path.join(BASE_PATH,"06_crames_v_matrix.csv")
OUTPUT_PAIR = os.path.join(BASE_PATH,"06_crames_v_paris.csv")
OUTPUT_DIR=os.path.join(BASE_PATH,"relationship_tables")
os.makedirs(OUTPUT_DIR,exist_ok=True)

print("\n[06 전체 조합 결과 불러오기]")
df=pd.read_csv(INPUT_FILE,encoding="utf-8-sig")
print(f"행 수: {len(df):,}")
print(f"컬럼 수 : {len(df.columns)}")
print(df.columns.tolist())

variables = ["장애유형","휠체어","시간구간","요일구분","거리구간","공간이동"]
print("\n"+"="*60)
print("분석 변수 확인")
print("="*60)
for col in variables:
    print(f"{col}: "
          f"{df[col].nunique(dropna=False):,}개 범주")

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

def weighted_crosstab(data,row_var,col_var,weight_col="건수"):
    table = pd.pivot_table(data,index=row_var,columns=col_var,values=weight_col,aggfunc="sum",fill_value=0,observed=False)
    return table

def cramers_v_from_table(table):
    observed = table.to_numpy(dtype=float)
    n=observed.sum()
    if n == 0:
        return np.nan
    row_sum = observed.sum(axis=1,keepdims= True)
    col_sum = observed.sum(axis=0,keepdims = True)
    expected=(row_sum@col_sum/n)
    valid = expected > 0
    chi2 = (((observed-expected)**2)/np.where(valid,expected,1))
    chi2=chi2[valid].sum()
    r,k = observed.shape
    denominator = min(r-1,k-1)
    if denominator <= 0:
        return np.nan
    v=np.sqrt(chi2/(n*denominator))
    return float(v)

print("\n"+"="*60)
print("6개 변수 Cramer's V 계산")
print("="*60)
matrix = pd.DataFrame(index = variables, columns=variables,dtype = float)
pair_results =[]
for var1 in variables:
    for var2 in variables:
        if var1 == var2:
            matrix.loc[var1,var2]=1.0
            continue
        table = weighted_crosstab(df,var1,var2)
        v=cramers_v_from_table(table)
        matrix.loc[var1,var2]=v

for i in range(len(variables)):
    for j in range(i+1,len(variables)):
        var1 = variables[i]
        var2 = variables[j]
        v=matrix.loc[var1,var2]
        pair_results.append({"변수1":var1,"변수2":var2,"Cramers_V":v})
pair_df=pd.DataFrame(pair_results)
pair_df=pair_df.sort_values("Cramers_V",ascending=False).reset_index(drop=True)
pair_df["순위"]=range(1,len(pair_df)+1)

print("\n[Cramer's V 관계행렬]")
print(matrix.round(4).to_string())
print("\n"+"="*60)
print("변수쌍 관계 강도 순위")
print("="*60)
print(pair_df.round(4).to_string(index=False))

important_pairs=[("거리구간","공간이동"),("시간구간","거리구간"),("시간구간","공간이동"),("요일구분","시간구간"),("장애유형","시간구간"),("장애유형","거리구간"),("장애유형","공간이동"),("장애유형","요일구분"),("휠체어","시간구간"),("휠체어","거리구간"),("휠체어","공간이동"),("휠체어","요일구분"),("장애유형","휠체어")]
print("\n"+"="*60)
print("주요 교차분포 저장")
print("="*60)
for var1,var2 in important_pairs:
    table = weighted_crosstab(df,var1,var2)
    row_pct = table.div(table.sum(axis=0)*100)
    count_file = os.path.join(OUTPUT_DIR,f"{var1}_{var2}_건수.csv")
    pct_file = os.path.join(OUTPUT_DIR,f"{var1}_{var2}행비율.csv")
    table.to_csv(count_file,encoding="utf-8-sig")
    row_pct.round(4).to_csv(pct_file,encoding="utf-8-sig")
    print(f"{var1}x{var2}저장 완료")

print("\n"+"="*60)
print("장애유형 관련 관계")
print("="*60)
disability_result = pair_df[(pair_df["변수1"]=="장애유형")|(pair_df["변수2"]=="장애유형")]
print(disability_result.round(4).to_string(index=False))
print("\n"+"="*60)
print("휠체어 관련 관계")
print("="*60)
wheelchair_result= pair_df[(pair_df["변수1"]=="휠체어")|(pair_df["변수2"]=="휠체어")]
print(wheelchair_result.round(4).to_string(index=False))

matrix.round(6).to_csv(OUTPUT_MATRIX,encoding="utf-8-sig")
pair_df.round(6).to_csv(OUTPUT_PAIR,index=False,encoding="utf-8-sig")
print("\n"+"="*60)
print("결과 저장")
print("="*60)
print(f"Cramer's V 행렬: \n"
      f"{OUTPUT_MATRIX}")
print(f"\n변수쌍 순위:\n"
      f"{OUTPUT_PAIR}")
print(f"\n교차표 폴더:\n"
      f"{OUTPUT_DIR}")

print("\n"+"="*60)
print("08 페르소나 핵심변수 관계 분석 완료")
print("="*60)
print("""이번 단계의 목적
      - 장애유형과 휠체어를 제거하기 위한 분석이 아님
      - 6개 핵심변수가 서로 어떻게 연결되는지 확인
      - 행동 아키타입 구조를 데이터 기반으로 결정하기 위한 진단
      특히 확인할 것
      1. 거리구간 x 거리이동
      2. 시간구간 x 거리구간
      3. 장애유형 x 시간/거리/공간이동
      4. 휠체어 x 시간/거리/공간이동
      5. 장애유형 x 휠체어
      아직 하지 않은 것
      - 변수 제거
      - 군집화
      - 최종 페르소나 선정
      - Synthetic ID 생성
      - 층화추출
      - 가상설문""")
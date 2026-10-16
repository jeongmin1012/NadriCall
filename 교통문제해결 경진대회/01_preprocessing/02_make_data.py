import os
import pandas as pd
from config import files, encodings

SAVE_PATH = r"E:\박정민\data"
os.makedirs(SAVE_PATH,exist_ok=True)
RENAME_MAP = {"휠체어유무":"휠체어 유무","픽업거리(m)":"픽업거리","이동거리(m)":"이동거리"}
COMMON_COLS=["접수번호","상태","접수일시","배차일시","탑승시간","하차시간","목적지1","장애유형","호출장소1","픽업거리","이동거리","휠체어 유무"]

def read_csv_auto(path):
    for enc in encodings:
        try:
            df=pd.read_csv(path,encoding=enc,on_bad_lines="skip")
            return df,enc
        except UnicodeDecodeError:
            continue
    return None,None

all_data=[]
for name,path in files.items():
    print("\n"+"="*60)
    print(f"{name} 처리 시작")
    print("="*60)
    df,used_encoding=read_csv_auto(path)
    if df is None:
        print(f"{name} 읽기 실패")
        continue
    print(f"원본: "
          f"{len(df):,}행"
          f"{len(df.columns)}개 컬럼")
    print(f"인코딩: "
          f"{used_encoding}")
    df.columns=(df.columns.astype(str).str.strip())
    unnamed_cols=[col for col in df.columns
                  if col.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    df=df.rename(columns=RENAME_MAP)
    available_cols=[col for col in COMMON_COLS
                    if col in df.columns]
    missing_cols=[col for col in COMMON_COLS
                  if col not in df.columns]
    if missing_cols:
        print(f"없는 컬럼: "
              f"{missing_cols}")
    df = df[available_cols].copy()
    year=int(str(name)[:4])
    df.insert(0,"연도",year)
    all_data.append(df)

all_df=pd.concat(all_data,ignore_index=True)
output_file=os.path.join(SAVE_PATH,"nadricall_2022_2025.csv")
all_df.to_csv(output_file,index=False,encoding="utf-8-sig")
print("\n"+"="*60)
print("4개년 통합 완료")
print("="*60)
print(f"전체 행 수 :"
      f"{len(all_df):,}")
print(f"전체 컬럼 수: "
      f"{len(all_df.columns)}")
print("\n연도별 행 수: ")
print(all_df["연도"].value_counts().sort_index())
print(f"\n저장 위치: "
      f"{output_file}")
print("="*60)
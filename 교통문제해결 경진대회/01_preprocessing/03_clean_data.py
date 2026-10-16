import os
import re
import pandas as pd

DATA_PATH = r"E:\박정민\data"
INPUT_FILE = os.path.join(DATA_PATH,"nadricall_2022_2025.csv")
SAVE_PATH = os.path.join(DATA_PATH,"clean")
os.makedirs(SAVE_PATH,exist_ok=True)

print("="*60)
print("통합 데이터 불러오기")
print("="*60)
df=pd.read_csv(INPUT_FILE,encoding="utf-8-sig",low_memory=False,on_bad_lines="skip")
print(f"행 수: {len(df):,}")
print(f"컬럼 수: {len(df.columns)}")
print()


df["연도"]=pd.to_numeric(df["연도"],errors="coerce").astype("Int64")
if "접수번호" in df.columns:
    df["접수번호"] = (df["접수번호"].astype("string").str.replace(r"\.0$","",regex=True).str.strip())
string_cols=["상태","징애유형","휠체어 유무","목적지1","호출장소1"]
for col in string_cols:
    if col in df.columns:
        df[col]=(df[col].astype("string").str.strip())

if "상태" in df.columns:
    df["상태"]=df["상태"].replace({"고객하차":"하차"})
if "장애유형" in df.columns:
    df["장애유형"]=df["장애유형"].replace({"장루.요루":"장루 및 요루","장루및요루":"장루 및 요루","장루및 요루":"장루 및 요루"})
if "휠체어 유무" in df.columns:
    df["휠체어 유무"] = df["휠체어 유무"].replace({"유":"휠체어","무":"비휠체어"})
for col in ["픽업거리","이동거리"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col],errors="coerce")

TIME_COLS={"접수일시":"접수","배차일시":"배차","탑승시간":"탑승","하차시간":"하차"}
weekday_map={0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
def parse_datatime_column(series):
    raw=(series.astype("string").str.strip())
    raw = raw.str.replace(r"\s*[월화수목금토일]요일$","",regex=True)
    raw=raw.str.replace(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일",r"\1-\2-\3",regex=True)
    converted = pd.to_datetime(raw,errors="coerce",format="mixed")
    return converted
for original_col, prefix in TIME_COLS.items():
    if original_col not in df.columns:
        continue
    print(f"[{original_col}처리]")
    converted = parse_datatime_column(df[original_col])
    df[original_col]=converted
    make_1970 = (converted.dt.year == 1970).fillna(False)
    df[f"{prefix}_1970여부"]=(make_1970.astype("Int64"))
    print(f"1970년 값: "
          f"{int(make_1970.sum()):,}건")
    df[f"{prefix}날짜"]=(converted.dt.strftime("%Y-%m-%d").astype("string"))
    df[f"{prefix}월"]=(converted.dt.month.astype("Int64"))
    df[f"{prefix}일"]=(converted.dt.day.astype("Int64"))
    df[f"{prefix}요일"]=(converted.dt.dayofweek.map(weekday_map).astype("string"))
    df[f"{prefix}시간대"]=(converted.dt.hour.astype("Int64"))
    df[f"{prefix}평일주말"]=(df[f"{prefix}요일"].map({"월":"평일","화":"평일","수":"평일","목":"평일","금":"평일","토":"주말","일":"주말"}).astype("string"))
    invalid_time_cols=[f"{prefix}날짜",f"{prefix}월",f"{prefix}일",f"{prefix}요일",f"{prefix}평일주말",f"{prefix}시간대"]
    for col in invalid_time_cols:
        df.loc[make_1970,col]=pd.NA
    mask_2025=(df["연도"].eq(2025).fillna(False))
    df.loc[mask_2025,f"{prefix}시간대"]=pd.NA
    print()

DAEGU_CLEAR=["대구광역시","달서구","수성구","달성군","군위군"]
DAEGU_CALL=["중구","동구","서구","남구","북구","수성구","달서구","달성군","군위군"]
GYEONGBUK=["포항시","경주시","김천시","안동시","구미시","영주시","영천시","상주시","문경시","경산시","의성군","청송군","영양군","영덕군","청도군","고령군","성주군","칠곡군","예천군","봉화군","울진군","울릉군"]
GYEONGNAM=["경상남도","경남","창녕군"]
DAEGU_GUGUN=["중구","동구","서구","남구","북구","수성구","달서구","달성군","군위군"]
REGION_PREFIXRES = ["대구광역시","경상북도","경상남도","대구","경북","경남"]

def clean_address_text(address):
    if pd.isna(address):
        return None
    text=str(address).strip()
    if (text=="" or text.lower()=="nan" or text.lower()=="none"):
        return None
    text=re.sub(r"\s+"," ",text).strip()
    return text
def remove_region_prefix(address):
    text=clean_address_text(address)
    if text is None:
        return ""
    for prefix in REGION_PREFIXRES:
        if text.startswith(prefix):
            text=text[len(prefix):
                      ].strip()
            break
    return text
def classify_region(address):
    text=clean_address_text(address)
    if text is None:
        return "미상"
    if re.fullmatch(r"\d+",text):
        return "주소코드"
    if (text.startswith("대구광역시") or text.startswith("대구")):
        return "대구"
    if any (area in text for area in DAEGU_GUGUN):
        return "대구"
    if (text.startswith("경상북도") or text.startswith("경북")):
        return "경북"
    if any(area in text for area in GYEONGBUK):
        return "경북"
    if(text.startswith("경상남도")or text.startswith("경남")):
        return "경남"
    if any(area in text for area in GYEONGNAM):
        return "경남"
    return "기타"

def classify_call_region(address):
    text = clean_address_text(address)
    if text is None:
        return "미상"
    if re.fullmatch(r"\d+",text):
        return "주소코드"
    if(text.startswith("대구광역시") or text.startswith("대구")):
        return "대구"
    if any(area in text for area in DAEGU_GUGUN):
        return "대구"
    if (text.startswith("경상북도") or text.startswith("경북")):
        return "경북"
    if any(area in text for area in GYEONGBUK):
        return "경북"
    if(text.startswith("경상남도")or text.startswith("경남")):
        return "경남"
    if any(area in text for area in GYEONGNAM):
        return "경남"
    return "기타"

def extract_gugun(address):
    text=clean_address_text(address)
    if text is None:
        return pd.NA
    if re.fullmatch(r"\d+",text):
        return pd.NA
    text = remove_region_prefix(text)
    if text == "":
        return pd.NA
    for gugun in DAEGU_GUGUN:
        if gugun in text:
            return gugun
    for gugun in GYEONGBUK:
        if gugun in text:
            return gugun
    for gugun in GYEONGNAM:
        if gugun in text:
            return gugun
    match = re.search(r"([가-힣0-9]+(?:시|구|군))",text)
    if match:
        result = match.group(1)
        invalid_values={"대구","경북","경남","대구광역시","경상북도","경상남도"}
        if result not in invalid_values:
            return result
    return pd.NA

def extract_dong(address):
    text=clean_address_text(address)
    if text is None:
        return pd.NA
    if re.fullmatch(r"\d+",text):
        return pd.NA
    text=remove_region_prefix(text)
    if text == "":
        return pd.NA
    gugun = extract_gugun(address)
    if pd.notna(gugun):
        gugun_text=str(gugun)
        idx=text.find(gugun_text)
        if idx>=0:
            text=(text[idx + len(gugun_text):].strip())
    if text == "":
        return pd.NA
    match=re.search(r"([가-힣0-9]+(?:동|읍|면))",text)
    if match:
        return match.group(1)
    return pd.NA

def extract_etc(address,gugun,dong):
    text=clean_address_text(address)
    if text is None:
        return pd.NA
    if re.fullmatch(r"\d+",text):
        return pd.NA
    text=remove_region_prefix(text)
    if text == "":
        return pd.NA
    if pd.notna(gugun):
        gugun_text=str(gugun)
        idx=text.find(gugun_text)
        if idx >= 0:
            text=(text[idx + len(gugun_text):].strip())
    if pd.notna(dong):
        dong_text=str(dong)
        idx=text.find(dong_text)
        if idx >- 0:
            text=(text[idx + len(dong_text):].strip())
    text = re.sub(r"^[\-/\s]+","",text).strip()
    if text == "":
        return pd.NA
    return text

print("="*60)
print("목적지 주소 분리")
print("="*60)
df["목적지_지역구분"]=(df["목적지1"].apply(classify_region).astype("string"))
df["목적지_구군"]=(df["목적지1"].apply(extract_gugun).astype("string"))
df["목적지_동읍면"]=(df["목적지1"].apply(extract_dong).astype("string"))
df["목적지_기타"]=(df.apply(lambda row: extract_etc(row["목적지1"],row["목적지_구군"],row["목적지_동읍면"]),axis=1).astype("string"))

print("\n"+"="*60)
print("호출장소 주소 분리")
print("="*60)
df["호출_지역구분"]=(df["호출장소1"].apply(classify_call_region).astype("string"))
df["호출_구군"]=(df["호출장소1"].apply(extract_gugun).astype("string"))
df["호출_동읍면"]=(df["호출장소1"].apply(extract_dong).astype("string"))
df["호출_기타"]=(df.apply(lambda row: extract_etc(row["호출장소1"],row["호출_구군"],row["호출_동읍면"]),axis=1).astype("string"))

if ("접수일시" in df.columns and "배차일시" in df.columns):
    df["이상_배차전접수"]=(df["배차일시"]<df["접수일시"]).astype("Int64")
if("배차일시" in df.columns and "탑승시간" in df.columns):
    df["이상_탑승전배차"]=(df["탑승시간"]<df["배차일시"]).astype("Int64")
if ("탑승시간" in df.columns and "하차시간" in df.columns):
    df["이상_하차전탑승"]=(df["하차시간"]<df["탑승시간"]).astype("Int64")
mask_2025=(df["연도"].eq(2025).fillna(False))
time_error_cols=["이상_배차전접수","이상_탑승전배차","이상_하차전탑승"]
for col in time_error_cols:
    if col in df.columns:
        df.loc[mask_2025,col]=pd.NA
    flag_cols=["접수_1970여부","배차_1970여부","탑승_1970여부","하차_1970_여부"]
    existing_flags=[col for col in flag_cols if col in df.columns]
    if existing_flags:
        mask_any_1970=(df[existing_flags].fillna(0).eq(1).any(axis=1))
        for col in time_error_cols:
            if col in df.columns:
                df.loc[mask_any_1970,col]=pd.NA

df["완료이동여부"]=df["상태"].eq("하차").fillna(False).astype("Int64")

summary=[]
for col in df.columns:
    summary.append({"column":col,"dtype":str(df[col].dtype),"missing":int(df[col].isna().sum()),"missing_rate":round(df[col].isna().mean()*100,4),"unique":int(df[col].nunique(dropna=True))})
summary_df=pd.DataFrame(summary)
summary_file=os.path.join(SAVE_PATH,"clean_schema_summary.csv")
summary_df.to_csv(summary_file,index=False,encoding="utf-8-sig")

print("\n"+"="*60)
print("연도별 clean 파일 저장")
print("="*60)
for year in sorted(df["연도"].dropna().unique()):
    year=int(year)
    year_df=df[df["연도"]==year].copy()
    output_file=os.path.join(SAVE_PATH,f"nadricall_{year}_clean.csv")
    year_df.to_csv(output_file,index=False,encoding="utf-8-sig")
    print(f"{year}: "
          f"{len(year_df):,}행 저장")

all_output=os.path.join(SAVE_PATH,"nadricall_2022_2025_clean.csv")
df.to_csv(all_output,index=False,encoding="utf-8-sig")

print("\n"+"="*60)
print("데이터 정리 완료 - 최종 검증")
print("="*60)
print(f"\n전체 행 수: "
      f"{len(df):,}")
print(f"전체 컬럼 수: "
      f"{len(df.columns)}")
print("\n[연도별 건수]")
print(df["연도"].value_counts().sort_index())
print("\n[상태별 건수]")
print(df["상태"].value_counts(dropna=False))
print("\n[연도 x 상태]")
print(pd.crosstab(df["연도"],df["상태"]))
print("\n[장애유형]")
print(df["장애유형"].value_counts(dropna=False))
print("\n[휠체어 유무]")
print(df["휠체어 유무"].value_counts(dropna=False))
print("\n[완료 이동 여부]")
print(df["완료이동여부"].value_counts(dropna=False))
print("\n[목적지 지역 구분]")
print(df["목적지_지역구분"].value_counts(dropna=False))
print("\n[호출장소 지역 구분]")
print(df["호출_지역구분"].value_counts(dropna=False))
region_df=df[df["연도"]<=2024].copy()
move_df=region_df[(region_df["상태"]=="하차")&(region_df["호출_지역구분"]!="미상")&(region_df["목적지_지역구분"]!="미상")].copy()
daegu_move=move_df[(move_df["호출_지역구분"]=="대구")&(move_df["목적지_지역구분"]=="대구")].copy()
print("\n[2022~2024 대구 -> 대구 완료 이동]")
print(f"{len(daegu_move):,}건")
print("\n[대구 완료 이동 호출 구군 TOP 10]")
print(daegu_move["호출_구군"].value_counts(dropna=False).head(10))
print("\n[대구 완료 이동 목적지 구군 TOP 10]")
print(daegu_move["목적지_구군"].value_counts(dropna=False).head(10))
if "탑승_1970여부" in df.columns:
    print("\n[상태 x 탑승 1970]")
    print(pd.crosstab(df["상태"],df["탑승_1970여부"],margins=True))
if "하차_1970여부" in df.columns:
    print("\n[상태 x 하차 1970]")
    print(pd.crosstab(df["상태"],df["하차_1970여부"],margins=True))
if("접수_1970여부" in df.columns and "접수시간대" in df.columns):
    print("\n[접수 1970 기록의 접수시간대 확인]")
    print(df.loc[df["접수_1970여부"]==1,"접수시간대"].value_counts(dropna=False))
print("\n[2025 접수 날짜 확인]")
print(df.loc[df["연도"]==2025,["접수일시","접수날짜","접수월","접수요일","접수평일주말","접수시간대"]].head(10).to_string(index=False))
print("\n[2025 접수시간대 결측 확인]")
print(df.loc[df["연도"]==2025,"접수시간대"].value_counts(dropna=False))
print("\n[2025 목적지 지역구분]")
print(df.loc[df["연도"]==2025,"목적지_지역구분"].value_counts(dropna=False))
distance_df=df[df["상태"]=="하차"].copy()
distance_df["이동거리"]=(pd.to_numeric(distance_df["이동거리"],errors="coerce"))
distance_df=(distance_df.dropna(subset=["이동거리"]))

distance_df=df[df["상태"]=="하차"].copy()
distance_df["이동거리"]=pd.to_numeric(distance_df["이동거리"],errors="coerce")
distance=distance_df["이동거리"].dropna()
print("\n[완료건 이동거리 통계]")
print(f"count : {distance.count():,}")
print(f"mean : {distance.mean():.2f}")
print(f"std : {distance.std():.2f}")
print(f"min : {distance.min():.2f}")
print(f"25% : {distance.quantile(0.25):.2f}")
print(f"50% : {distance.quantile(0.50):.2f}")
print(f"75% : {distance.quantile(0.75):.2f}")
print(f"90% : {distance.quantile(0.90):.2f}")
print(f"95% : {distance.quantile(0.95):.2f}")
print(f"99% : {distance.quantile(0.99):.2f}")
print(f"max : {distance.max():.2f}")

print("\n"+"="*60)
print(f"전체 clean 파일:{all_output}")
print(f"\n스키마 요약:\n{summary_file}")
print("="*60)
"""나드리콜 A/N 가상 설문 통합 실행 코드.

이 파일을 한 번 실행하면 A 실험과 N 실험을 차례로 수행한다.
10,000명 최종 실행일 때는 두 실험이 끝난 뒤 ``compare_h4.py``도
자동으로 실행한다.

2021년 H4 실측분포는 정답 유출을 막기 위해 이 코드에 넣지 않았다.
실측분포는 A와 N 생성이 모두 끝난 뒤 비교 코드에서만 사용한다.
"""

from getpass import getpass
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from openai import OpenAI
except ImportError as error:
    raise ImportError(
        "openai 패키지가 없습니다. 프로젝트 폴더에서 "
        "'python -m pip install -r requirements.txt'를 실행하세요."
    ) from error


def _load_dotenv_file(path: Path) -> None:
    """python-dotenv 없이도 .env를 읽을 수 있게 한다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )


_load_dotenv_file(Path(__file__).resolve().parent / ".env")


# ============================================================
# 0. 사용자 설정
# ============================================================

# 실행 순서다. A가 끝나면 N을 자동으로 실행한다.
# 현재는 A만 재보정 실행한다.
RUN_MODES = ("A",)

# 10,000명 최종 실행 후 비교 코드까지 자동으로 실행할지 정한다.
RUN_COMPARISON = False

# API 요청 한 번에 함께 보낼 응답자 수다.
# 선택한 모델의 입력·출력 한도가 작다면 이 값을 줄인다.
BATCH_SIZE = 20

# 10,000명 전체를 실행할 때는 None을 사용한다.
# 시험 실행만 할 때 100 같은 정수를 넣으며, 시험 결과는 최종평가에 쓰지 않는다.
# 일부만 시험 실행 (API 비용·시간 절약). 최종 1만 명은 None.
TEST_LIMIT = None

MAX_RETRY = 5
PROMPT_VERSION = "2026-10-12-A-v7-temperament-iterate"

# 실측 분포 복사(순위 보정)는 사용하지 않는다.
H4_RANK_CALIBRATION = False
H4_SOFT_CONSISTENCY = False

# H4만 동일 응답자 H2·H3·이동특성·성향으로 재추론 (실측 건수 배정 아님)
H4_SURVEY_CONSISTENCY = True

BASE_PATH = Path(__file__).resolve().parent
RESULT_DIR = BASE_PATH / "survey_results"
CACHE_DIR = RESULT_DIR / "cache"

RESULT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# 원래 파일명과 ``16_synthetic_population_10000.csv``처럼
# 내려받으면서 번호가 붙은 파일명을 모두 인식한다.
INPUT_STEMS = {
    "A": "16_synthetic_population_10000",
    "N": "22_nemotron_control_10000",
}

FACT_COLS = [
    "장애유형",
    "휠체어",
    "시간구간",
    "요일구분",
    "거리구간",
    "공간이동",
]

QUESTION_COLS = [
    "H2_1",
    "H2_2",
    "H2_3",
    "H2_4",
    "H2_5",
    "H3",
    "H4",
]

SCORE_LABELS = {
    1: "매우 불만족",
    2: "불만족",
    3: "약간 불만족",
    4: "보통",
    5: "약간 만족",
    6: "만족",
    7: "매우 만족",
}


# ============================================================
# 1. 설문지와 실험 프롬프트
# ============================================================

SURVEY_TEXT = """
[설문]

최근 1년 이내 대구광역시 특별교통수단 나드리콜을 이용한 경험이
있다고 가정하고 다음 문항에 각각 1~7점으로 응답한다.

H2. 특별교통수단 이용 평가

H2_1. 승무원의 친절성
H2_2. 예약 또는 배차의 편리성
H2_3. 차량 내부의 청결성
H2_4. 안전운전
H2_5. 이용요금

H2 척도:
1 매우 불만족
2 불만족
3 약간 불만족
4 보통
5 약간 만족
6 만족
7 매우 만족

H3. 특별교통수단 이용 전 기대수준

1 전혀 기대하지 않았다
2 기대하지 않았다
3 기대하지 않는 편이었다
4 보통이었다
5 기대하는 편이었다
6 기대했다
7 매우 기대했다

H4. 특별교통수단 전반적 만족도

1 매우 불만족
2 불만족
3 약간 불만족
4 보통
5 약간 만족
6 만족
7 매우 만족

H2의 다섯 문항은 각각 별도로 판단한다.

H3는 이용 전 기대수준이므로
H2 또는 H4와 같은 값일 필요는 없다.

H4는 전반적 만족도에 대한 종합판단이다.
H2의 평가와 관련될 수 있지만
다섯 문항의 단순 평균으로 계산하지 않는다.
""".strip()

COMMON_RULES = """
당신은 대구광역시 특별교통수단 나드리콜 이용자의 설문응답을
예측하는 가상 설문조사 시뮬레이터이다.

이 작업은 실제로 발생한 서비스 사건을 복원하는 작업이 아니다.

제공된 응답자 정보를 바탕으로,
해당 응답자가 설문에 참여했다면 각 문항에 선택할 법한
주관적인 응답을 일관된 기준으로 판단하는 작업이다.


[기본 원칙]

1. 각 응답자를 서로 독립적으로 판단한다.

같은 배치에 포함된 다른 응답자의 정보나 점수를 참고하지 않는다.
전체 응답분포나 특정 점수의 사용 횟수를 고려하지 않는다.


2. 웹 검색과 외부자료를 사용하지 않는다.

특정 연도의 공식 조사표나 목표 점수 비율표를
그대로 복사해 맞추려고 하지 않는다.
다만 아래 [현실성 보정]의 일반적인 경향은 반영한다.


3. 입력에 없는 구체적인 서비스 사건을 만들지 않는다.

다음과 같은 내용을 새로운 사실처럼 만들어서는 안 된다.

- 실제 대기시간
- 구체적인 배차 성공 또는 실패
- 승무원과의 구체적인 대화나 행동
- 차량의 실제 청결 상태
- 사고 또는 위험상황
- 실제 이용요금 금액
- 입력에 없는 구체적인 이동 목적
- 입력에 없는 개인의 질병이나 생활경험


4. 구체적인 사건을 만들지 않는 범위에서
제공된 정보와 일반적으로 예상할 수 있는 관계를 종합할 수 있다.

다음과 같은 내용을 추론할 수 있다.

- 이동환경
- 이동부담
- 접근성 요구
- 특별교통수단에 대한 필요성
- 서비스의 이용가치
- 이용 전 기대수준
- 편의성에 대한 평가기준
- 만족 또는 불만족을 표현하는 성향


5. 이동부담과 서비스 이용가치는 서로 다른 평가축이다.

이동부담이 크다는 이유만으로
서비스에 낮은 점수를 부여하지 않는다.

서비스 의존성이나 이용가치가 높다는 이유만으로
서비스에 높은 점수를 부여하지 않는다.

이동부담과 이용가치는 동시에 존재할 수 있으며,
항상 같은 크기로 상쇄되는 것은 아니다.


[일관된 판단 절차]

모든 응답자와 모든 문항에 다음 판단 절차를 동일하게 적용한다.


1단계. 관련 정보 구분

해당 문항과 직접 또는 간접적으로 관련된 입력정보를 구분한다.

입력정보를 다음 세 방향으로 나누어 검토한다.

- 이동부담 또는 접근성 요구를 높이는 방향
- 서비스의 필요성 또는 이용가치를 높이는 방향
- 어느 방향과도 뚜렷하게 연결되지 않는 중립적 정보


2단계. 평가 방향 판단

한 가지 특성만으로 점수를 결정하지 않는다.

여러 특성이 같은 방향을 가리키는지,
서로 반대되는 방향을 가리키는지,
전체 프로필에서 어떤 평가가 가장 그럴듯한지를 판단한다.

내부적으로 각 문항의 평가 방향을 다음 중 하나로 구분한다.

- 부정 방향
- 중립
- 긍정 방향


3단계. 자유로운 가치판단과 점수 선택

제공된 정보 전체를 하나의 응답자 프로필로 받아들이고,
그 응답자가 실제 설문에 참여했다면 선택할 법한 점수를 판단한다.

각 점수를 선택하기 위해 특정 개수의 근거나
사전에 정해진 조건을 충족할 필요는 없다.

문항과 직접 연결되는 정보가 적더라도
자동으로 4점을 선택하지 않는다.

이 경우에도 응답자의 이동환경, 접근성 요구,
서비스 필요성, 이용가치, 기대수준과 평가성향을 종합하여
1점부터 7점 사이에서 가장 그럴듯한 점수를 자유롭게 선택한다.

4점은 정보가 부족할 때 사용하는 기본값이 아니다.
응답자의 주관적인 평가가 실제로 중립에 가깝다고 판단될 때 선택한다.

1점과 7점을 포함한 모든 점수는 동등하게 선택 가능한 응답이다.
제공된 프로필 전체에서 강한 만족 또는 불만족이 가장 그럴듯하다면
구체적인 서비스 사건을 만들지 않고도 1점 또는 7점을 선택할 수 있다.


4단계. 최종 일관성 확인

최종 점수를 작성하기 전에 다음을 확인한다.

- 입력에 없는 구체적인 사건을 만들지 않았는가
- 특정 장애유형이나 하나의 특성만으로 점수를 결정하지 않았는가
- 정보가 부족하다는 이유만으로 모든 문항을 4점으로 맞추지 않았는가
- 여러 문항을 기계적으로 같은 점수로 작성하지 않았는가
- H3를 만족도 점수처럼 판단하지 않았는가
- H4를 H2 다섯 문항의 단순 평균으로 계산하지 않았는가


[현실성 보정 — LLM 편향 교정]

언어모델은 H4에서 '5·6점'에 과도하게 몰리고 1·2·7점을 거의 쓰지 않는 경향이 있다.
이 편향을 교정하되, 실측표·목표비율을 맞추려 하지 않는다.

특별교통수단 이용자 설문의 H4(전반적 만족도)에서는 대체로:

- 4점(보통)이 가장 흔한 응답 중 하나다.
- 1~3점(불만족 쪽)과 5~7점(만족 쪽)이 모두 존재한다.
- 5~6점만으로 과반을 채우지 않는다.
- 1·2·7점은 드물지 않으며, 응답자 성향·H2·H3와 맞을 때 선택한다.

각 응답자의 `응답_성향`·`H4_판단힌트`를 H4에 반영한다.
한 배치(20명)에서 H4가 5 또는 6만 연속되지 않게 한다.


[점수 사용 원칙]

1점부터 7점까지 모든 점수를 사용할 수 있다.

특정 점수를 기본값으로 삼지 않는다.
5점·6점을 기본값처럼 쓰지 않는다.
중앙값이나 극단값을 의도적으로 회피하지 않는다.

1점·2점 또는 6점·7점은
구체적인 서비스 사건이 확인된 경우에만
선택할 수 있는 점수가 아니다.

제공된 특성의 조합에서 추론되는
이동환경, 이동부담, 접근성 요구, 서비스 이용가치,
기대수준과 평가성향을 종합했을 때
한 방향의 주관적인 평가가 강하게 나타난다면
낮거나 높은 점수를 선택할 수 있다.

낮거나 높은 점수를 선택하더라도
배차 실패, 긴 대기시간, 승무원 행동, 사고,
차량 청결 상태와 같은 구체적인 사건이
실제로 발생했다고 가정해서는 안 된다.

긍정적 요인과 부정적 요인이 함께 존재한다고 해서
항상 기계적으로 상쇄하여 4점을 선택하지 않는다.

직접적인 서비스 사건이 제공되지 않았다는 이유만으로
응답자의 주관적 가치판단까지 제한하지 않는다.

구체적인 사실을 새로 만드는 것과
제공된 프로필을 바탕으로 주관적 평가를 추론하는 것은 다르다.

전자는 금지하지만, 후자는 이 가상설문의 핵심 과업이므로
모든 점수 범위에서 자유롭게 수행한다.


[문항별 판단 원칙]

H2의 다섯 문항은 서로 다른 서비스 영역이다.
기계적으로 같은 점수를 부여하지 않는다.

H2_1 승무원의 친절성,
H2_3 차량 내부의 청결성,
H2_4 안전운전처럼
제공된 정보와 직접 연결되지 않는 문항은
구체적인 사건을 새로 만들지 않는 범위에서 판단한다.

이 문항들은 접근성 요구, 기대수준,
서비스 의존성과 전반적인 평가성향을 종합하여
가상 응답자가 어떻게 평가했을지를 자유롭게 판단할 수 있다.

다만 이동정보 또는 인물정보를
실제 승무원 행동, 청결 상태나 안전운전 경험으로
바꾸어 해석해서는 안 된다.

H2_2 예약 또는 배차의 편리성은
제공된 정보 중 시간대, 요일, 이동거리,
공간이동과 접근성 요구를 상대적으로 더 관련 있게
고려할 수 있다.

H2_5 이용요금은 실제 요금 금액이나 소득을 만들지 않는다.
제공된 정보에서 추론할 수 있는 이동범위,
서비스 이용가치와 평가기준을 종합하여 자유롭게 판단한다.


[H3 판단 원칙]

H3는 서비스를 이용하기 전의 기대수준이다.

만족도인 H2 또는 H4와 같은 점수일 필요는 없다.

서비스 필요성, 접근성 요구, 생활환경,
이동 편의성에 대한 기준과 평가성향을 종합하여
이용 전에 어느 정도 기대했을지를 판단한다.

이용 후 만족도가 높다고 해서
이용 전 기대수준도 반드시 높았다고 판단하지 않는다.

이용 후 만족도가 낮다고 해서
이용 전 기대수준도 반드시 낮았다고 판단하지 않는다.


[H4 판단 원칙]

H4는 전반적 만족도에 대한 독립적인 종합판단이다.

H4를 판단할 때는 다음 요소를 함께 고려한다.

- H2의 다섯 평가영역에서 나타난 전체적인 방향
- 응답자에게 특별교통수단이 가지는 필요성과 이용가치
- 이동환경과 접근성 요구에서 예상되는 이용부담
- H3 기대수준과 응답자의 평가기준
- [현실성 보정]의 일반 경향

이 요소에는 고정된 우선순위나 계산식이 없다.
응답자 프로필 전체에서 가장 중요하게 작용할 것으로 판단되는
요인과 주관적인 평가 방향을 자유롭게 반영한다.

H4는 H2 다섯 문항의 산술평균으로 계산하지 않는다.

다만 특별한 판단 근거 없이
H2 전체의 평가 방향과 정반대가 되지 않도록 한다.

H4는 먼저 4점(보통)을 검토한 뒤, H2·H3·응답_성향에 따라 위·아래로 조정한다.
5·6을 LLM 기본값처럼 쓰지 않는다.

- H2 평균이 4 이하 또는 H2_2·H2_5가 낮으면 H4는 1~4(특히 2~3)를 검토한다.
- H2가 혼재하면 H4는 3~5(중심 4)를 검토한다.
- H2가 전반적으로 6 이상이고 H3도 높으면 H4는 5~7(7 포함)을 검토한다.

H4는 H2와 같은 값일 필요는 없지만, H2 전반과 정반대(예: H2≈6인데 H4=1)는 피한다.
응답_성향이 '엄격'이면 H4 상한을 낮추고, '긍정'이면 하한을 높인다.
'양극' 성향은 H2·H3가 분명할 때 1~2 또는 7을 검토한다.

이용가치가 높다는 이유만으로 높은 만족도를 주지 않고,
이동부담이 크다는 이유만으로 낮은 만족도를 주지 않는다.

두 요인 중 응답자의 종합평가에 더 크게 작용할 것으로
판단되는 방향과 강도를 반영한다.


[출력 원칙]

응답자 ID는 판단정보로 사용하지 않는다.

아무 근거 없는 무작위 값을 넣지 않는다.

공식 조사표의 점수 비율을 베끼지 않는다.
그러나 5~6점 과다 쏠림은 피하고 1~4점·7점도 충분히 사용한다.

내부적인 판단과정이나 설명은 출력하지 않는다.

모든 응답은 1부터 7 사이의 정수로 작성한다.
""".strip()

A_SYSTEM_PROMPT = f"""
{COMMON_RULES}

[실험 A: 나드리콜 이동행동 기반 합성인구]

이 실험에서 사용할 수 있는 개인 정보는
실제 나드리콜 완료 이동데이터에서 구성한
다음 6개 이동행동 정보뿐이다.

- 장애유형
- 휠체어 여부
- 시간구간
- 요일구분
- 거리구간
- 공간이동


[정보의 의미]

이 정보는 나드리콜 완료 이동에서 관찰된 행동특성이다.

이동이 완료되었다는 사실 자체는
만족 또는 불만족의 근거가 아니다.

나드리콜이 이동을 가능하게 했다는 이용가치 역시
높은 만족도를 자동으로 의미하지 않는다.

여섯 정보를 함께 고려하여 다음을 판단한다.

- 이동 과정에서 특별교통수단이 가지는 필요성
- 특별교통수단을 통해 얻을 수 있는 이용가치
- 시간대와 요일에 따른 이동환경 및 편의성에 대한 기준
- 이동거리와 구군 이동에 따른 활동범위와 이동부담
- 장애유형과 휠체어 여부에 따른 접근성 요구
- 여러 특성이 결합되었을 때 나타날 수 있는 평가 방향
- 제공된 행동특성에서 예상되는 서비스 평가기준


[특성 해석 원칙]

장거리 이동, 타 구군 이동,
비일상적인 시간대의 이동,
휠체어 이용과 같은 특성은
이동부담이나 접근성 요구를 높일 수 있다.

그러나 이러한 특성만으로
낮은 만족도를 부여하지 않는다.

반대로 서비스 의존성이 높거나
나드리콜을 통한 이동의 이용가치가 크다고 판단되더라도
그것만으로 높은 만족도를 부여하지 않는다.

이동부담과 서비스 이용가치를 각각 판단한 뒤,
두 방향 중 어느 쪽이 해당 문항의 평가에
더 크게 작용할지를 판단한다.

여러 특성이 같은 방향을 가리키는 경우에는
그 방향과 강도를 점수에 반영할 수 있다.

특성들이 서로 반대 방향을 가리키는 경우에도
자동으로 중립적인 점수를 선택하지 않는다.

응답자 프로필 전체에서 어떤 요인이 더 중요하게 작용할지 판단하여
긍정, 부정 또는 중립 중 가장 그럴듯한 방향을 자유롭게 선택한다.


[문항과 이동정보의 관계]

H2_2 예약 또는 배차의 편리성은
시간구간, 요일구분, 거리구간,
공간이동과 접근성 요구를
다른 문항보다 상대적으로 더 관련 있게 고려할 수 있다.

H2_1 승무원의 친절성,
H2_3 차량 내부의 청결성,
H2_4 안전운전은
이동정보와 직접 연결되지 않는 영역이다.

따라서 실제 친절·청결·안전 경험을 만들지 않는다.

접근성 요구, 기대수준,
서비스 의존성과 전반적인 평가성향을 종합하여
가상 응답자가 각 영역을 어떻게 평가했을지 자유롭게 판단할 수 있다.

H2_5 이용요금은 실제 요금이나 소득을 만들지 않는다.
이동거리, 활동범위, 서비스 이용가치와
평가기준을 종합하여 자유롭게 판단한다.


[금지사항]

특정 장애유형을 고정된 만족도 점수와 연결하지 않는다.

휠체어 이용자를 자동으로 불만족한 응답자로 판단하지 않는다.

비휠체어 이용자를 자동으로 만족한 응답자로 판단하지 않는다.

장거리 또는 타 구군 이동을
배차 실패나 긴 대기시간이 발생했다는 뜻으로 해석하지 않는다.

심야·새벽 또는 야간 이동을
위험하거나 불만족스러운 경험이 있었다는 뜻으로 해석하지 않는다.

성별, 정확한 연령, 직업, 소득, 가족관계,
추가 질병, 성격과 구체적인 개인 경험 등
입력되지 않은 개인특성을 새로 만들지 않는다.

제공된 여섯 이동특성의 조합만을 사용하여
가장 그럴듯한 주관적 응답을 판단한다.
""".strip()

N_SYSTEM_PROMPT = f"""
{COMMON_RULES}

[실험 N: 일반 Nemotron 대조군]

이 실험에서 사용할 수 있는 개인 정보는
Nemotron_Context뿐이다.

Nemotron_Context는 일반인의 인구통계,
가족관계, 생활환경, 생활방식과 성향을 표현한
일반 페르소나 정보이다.

이 정보는 실제 나드리콜 이동기록이나
실제 나드리콜 서비스 만족도 자료가 아니다.


[응답자 가정]

응답자는 설문 가정상
나드리콜을 이용할 수 있는 정당한 이용자이며,
최근 1년 이내 이용 경험이 있다고 가정한다.

이용자격의 구체적인 이유는 제공되지 않았으므로
새로 만들지 않는다.

현재 거주지역이 대구가 아니라는 이유로
점수를 낮추거나 나드리콜을 이용할 수 없다고 판단하지 않는다.


[사용 가능한 판단정보]

Nemotron_Context에서 나타나는
연령, 직업, 가족환경, 생활방식과 성향을 바탕으로
다음 내용을 판단할 수 있다.

- 이용 전 기대수준
- 이동 편의성에 대해 가질 수 있는 기준
- 서비스를 평가할 때 중요하게 여길 수 있는 요소
- 만족 또는 불만족을 표현하는 평가성향
- 평가를 강하게 또는 완만하게 표현할 가능성

이 정보는 응답자의 기대수준,
평가기준과 응답성향을 추론하고,
그 인물이 설문에 참여했다면 선택할 법한 점수를
자유롭게 판단하는 데 사용한다.

특정 나드리콜 서비스 경험이
실제로 좋거나 나빴다는 증거로 사용하지 않는다.


[성격과 생활환경 해석 원칙]

인물이 다정하다, 성실하다, 긍정적이다,
소박하다 또는 가족관계가 원만하다고
묘사되었다는 이유만으로
나드리콜 만족도를 높게 평가하지 않는다.

반대로 예민하다, 무뚝뚝하다, 고집이 있다,
불안하다 또는 생활이 어렵다고
묘사되었다는 이유만으로
나드리콜 만족도를 낮게 평가하지 않는다.

성격 표현은 서비스 경험의 증거가 아니다.

성격과 생활환경은 기대수준,
서비스 평가기준, 만족과 불만족을 표현하는 방식과 강도를
판단하는 데 종합적으로 고려할 수 있다.

인물의 전반적인 삶에 대한 긍정적 또는 부정적 인상과
특정 교통서비스에 대한 만족도를 구분한다.


[문항별 판단 원칙]

H2_1 승무원의 친절성,
H2_3 차량 내부의 청결성,
H2_4 안전운전은
Nemotron_Context만으로 실제 경험을 알 수 없는 영역이다.

따라서 구체적인 친절·청결·안전 경험을 만들지 않는다.

응답자의 기대수준,
서비스 평가기준과 평가성향을 종합하여
가상 응답자가 어떻게 평가했을지 자유롭게 판단한다.

H2_2 예약 또는 배차의 편리성은
생활방식, 일상 일정, 이동 편의성에 대한 기준과
평가성향을 종합하여 자유롭게 판단할 수 있다.

구체적인 대기시간이나 배차 실패를 만들지 않는다.

H2_5 이용요금은 실제 소득이나 요금 부담 경험을 만들지 않는다.

직업, 생활환경과 소비 또는 서비스 평가기준이
어떻게 연결될 수 있는지 종합하여 자유롭게 판단한다.

H3는 응답자의 생활환경,
편의성에 대한 기준과 일반적인 기대성향을 중심으로 판단한다.

H4는 인물의 삶 전반에 대한 긍정적·부정적 인상이 아니라
가정된 나드리콜 이용경험에 대한 종합평가로 판단한다.


[금지사항]

지역, 직업, 가족정보를
구체적인 나드리콜 이용경험으로 바꾸어 해석하지 않는다.

새로운 대기시간, 배차 실패, 승무원 응대,
차량 청결 또는 안전운전 경험을 만들지 않는다.

직업이나 학력만으로 만족도 점수를 결정하지 않는다.

연령만으로 기대수준이나 만족도를 결정하지 않는다.

가족형태나 혼인상태만으로
긍정적 또는 부정적 점수를 결정하지 않는다.

일반적인 긍정적 인물 묘사가
H2와 H4 전체를 높은 점수로 끌어올리지 않도록 한다.

일반적인 부정적 인물 묘사가
H2와 H4 전체를 낮은 점수로 끌어내리지 않도록 한다.

Nemotron_Context에서 확인되는
기대수준, 평가기준과 응답성향만을 사용하여
가장 그럴듯한 주관적 응답을 판단한다.
""".strip()

# ============================================================
# 2. 파일과 데이터 처리 함수
# ============================================================

def validate_settings():
    if not RUN_MODES:
        raise ValueError("RUN_MODES가 비어 있습니다.")

    invalid_modes = [mode for mode in RUN_MODES if mode not in {"A", "N"}]
    if invalid_modes:
        raise ValueError(f"지원하지 않는 실행 모드입니다: {invalid_modes}")

    if len(set(RUN_MODES)) != len(RUN_MODES):
        raise ValueError("RUN_MODES에 중복된 실험이 있습니다.")

    if not isinstance(BATCH_SIZE, int) or BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE는 1 이상의 정수여야 합니다.")

    if TEST_LIMIT is not None and (
        not isinstance(TEST_LIMIT, int) or TEST_LIMIT < 1
    ):
        raise ValueError("TEST_LIMIT는 None 또는 1 이상의 정수여야 합니다.")

    compare_file = BASE_PATH / "compare_h4.py"
    if TEST_LIMIT is None and RUN_COMPARISON and not compare_file.exists():
        raise FileNotFoundError(
            "최종 비교에 필요한 compare_h4.py가 없습니다. "
            "run_survey.py와 같은 폴더에 두세요."
        )


def resolve_input_file(mode):
    stem = INPUT_STEMS[mode]
    exact = BASE_PATH / f"{stem}.csv"

    if exact.exists():
        return exact

    matches = sorted(BASE_PATH.glob(f"{stem}*.csv"))

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"입력파일을 찾을 수 없습니다: {stem}.csv\n"
            "이 파이썬 파일과 CSV를 같은 폴더에 두세요."
        )

    names = "\n- ".join(path.name for path in matches)
    raise FileExistsError(
        f"후보 CSV가 여러 개입니다. 사용할 파일 하나만 남겨주세요.\n- {names}"
    )


def read_csv_auto(path):
    for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            data = pd.read_csv(path, encoding=encoding, low_memory=False)
            print(f"[불러오기 성공] {path.name} / {encoding}")
            return data
        except UnicodeDecodeError:
            continue

    raise ValueError(f"CSV 인코딩을 확인할 수 없습니다: {path}")


def prepare_input_data(mode):
    input_file = resolve_input_file(mode)
    data = read_csv_auto(input_file)

    if mode == "A":
        id_col = "Synthetic_ID"
        input_cols = FACT_COLS
        required = [id_col] + FACT_COLS
    else:
        id_col = "Nemotron_ID"
        input_cols = ["Nemotron_Context"]
        required = [id_col, "Nemotron_Context"]

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"필수 컬럼이 없습니다: {missing}")

    if data[id_col].isna().any():
        raise ValueError(f"{id_col}에 결측값이 있습니다.")

    if data[id_col].duplicated().any():
        raise ValueError(f"{id_col}에 중복값이 있습니다.")

    if data[input_cols].isna().any().any():
        counts = data[input_cols].isna().sum()
        counts = counts[counts > 0].to_dict()
        raise ValueError(f"LLM 입력 컬럼에 결측값이 있습니다: {counts}")

    if TEST_LIMIT is not None:
        data = data.head(TEST_LIMIT).copy()

    if TEST_LIMIT is None and len(data) != 10_000:
        raise ValueError(
            f"최종 실험 입력은 10,000행이어야 합니다. 현재: {len(data):,}행"
        )

    unique_input = data[input_cols].drop_duplicates().reset_index(drop=True)

    print(f"입력 행 수: {len(data):,}")
    print(f"{id_col} 고유값: {data[id_col].nunique():,}")
    print(f"고유 API 입력 수: {len(unique_input):,}")
    print(f"예상 API 요청 수: {(len(unique_input) + BATCH_SIZE - 1) // BATCH_SIZE:,}")

    return data, id_col, input_cols, unique_input


# ============================================================
# 3. 캐시 처리 함수
# ============================================================

def normalize_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_cache_key(mode, model, row, input_cols):
    payload = {
        "prompt_version": PROMPT_VERSION,
        "mode": mode,
        "model": model,
        "input": {column: normalize_value(row[column]) for column in input_cols},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_file_for_run(mode):
    suffix = "test" if TEST_LIMIT is not None else "final"
    return CACHE_DIR / f"{mode}_{suffix}_survey_cache.jsonl"


def load_cache(cache_file):
    cache = {}
    if not cache_file.exists():
        return cache

    with cache_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                cache[item["cache_key"]] = item["response"]
            except (json.JSONDecodeError, KeyError):
                print(f"[경고] 손상된 캐시 {line_number}행을 건너뜁니다.")

    print(f"[기존 캐시] {len(cache):,}개")
    return cache


def append_cache_records(cache_file, records):
    with cache_file.open("a", encoding="utf-8") as handle:
        for cache_key, response_data in records:
            record = {"cache_key": cache_key, "response": response_data}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ============================================================
# 4. API 배치 요청
# ============================================================

def response_temperament_hint(row) -> tuple[str, str]:
    payload = "|".join(normalize_value(row[column]) for column in FACT_COLS)
    bucket = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16) % 100

    if bucket < 28:
        return (
            "엄격",
            "서비스 아쉬움을 분명히 표현하는 편. H4는 1~4를 우선(2~3 흔함), "
            "H2가 높아도 6 이상은 신중히.",
        )
    if bucket < 68:
        return (
            "중립",
            "균형 잡힌 평가. H4는 3~5를 우선하며 전형적으로 4(보통). "
            "H2와 크게 어긋나지 않게.",
        )
    if bucket < 85:
        return (
            "긍정",
            "만족을 표현하는 편. H4는 4~6을 우선, H2가 전반적으로 높을 때 7 검토. "
            "H2가 낮으면 3~5로 조정.",
        )
    return (
        "양극",
        "만족·불만 중 하나가 분명할 때 극단 점수 가능. H2·H3가 낮으면 1~2, "
        "H2·H3가 높으면 6~7(7 포함) 검토.",
    )


def movement_burden_hint(row) -> str:
    burden = 0.0
    if normalize_value(row.get("휠체어")) == "휠체어":
        burden += 1.5
    distance = normalize_value(row.get("거리구간"))
    if any(token in distance for token in ("장", "20", "중장", "10~")):
        burden += 1.0
    space = normalize_value(row.get("공간이동"))
    if any(token in space for token in ("타", "구", "외")):
        burden += 0.8
    time_slot = normalize_value(row.get("시간구간"))
    if time_slot in {"저녁", "새벽", "심야", "야간", "밤"}:
        burden += 0.5

    if burden >= 2.5:
        return (
            "이동·접근 부담 신호: 높음. H4는 H2·H3와 함께 2~5를 중심으로 판단하고, "
            "극단(1·7)은 H2 전반이 매우 낮거나 높을 때만."
        )
    if burden >= 1.2:
        return (
            "이동·접근 부담 신호: 중간. H4는 3~5를 중심으로, "
            "H2_2·H2_5가 낮으면 2~4, 높으면 5~6도 검토."
        )
    return (
        "이동·접근 부담 신호: 낮음~보통. H4는 4~6을 중심으로, "
        "H2가 전반적으로 우수하면 7, 혼재하면 4~5."
    )


def build_batch_items(mode, batch):
    items = []
    key_to_row = {}

    for offset, (_, row) in enumerate(batch.iterrows(), start=1):
        agent_key = f"R{offset:03d}"
        key_to_row[agent_key] = row

        if mode == "A":
            temperament, temperament_hint = response_temperament_hint(row)
            item = {
                "agent_key": agent_key,
                **{column: normalize_value(row[column]) for column in FACT_COLS},
                "응답_성향": temperament,
                "응답_성향_설명": temperament_hint,
                "H4_판단힌트": movement_burden_hint(row),
            }
        else:
            item = {
                "agent_key": agent_key,
                "Nemotron_Context": normalize_value(row["Nemotron_Context"]),
            }

        items.append(item)

    return items, key_to_row


def make_batch_schema(agent_keys):
    response_properties = {
        "agent_key": {"type": "string", "enum": agent_keys},
        **{column: {"type": "integer", "minimum": 1, "maximum": 7}
           for column in QUESTION_COLS},
    }

    return {
        "type": "object",
        "properties": {
            "responses": {
                "type": "array",
                "minItems": len(agent_keys),
                "maxItems": len(agent_keys),
                "items": {
                    "type": "object",
                    "properties": response_properties,
                    "required": ["agent_key"] + QUESTION_COLS,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["responses"],
        "additionalProperties": False,
    }


def make_batch_user_prompt(items):
    return f"""
{SURVEY_TEXT}

[응답자 입력]
아래 JSON 배열의 각 응답자를 서로 독립적으로 판단하라.
각 agent_key에 대해 H2_1~H2_5, H3, H4를 모두 응답하라.
응답_성향·응답_성향_설명·H4_판단힌트는 H4 판단에만 참고한다.
배치 전체에서 H4가 5·6만 과도하게 나오지 않게 하고, 1~4·7도 성향에 맞으면 사용한다.
H2를 먼저 판단한 뒤 H3, 마지막에 H4를 H2·H3·성향과 일관되게 선택한다.

{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()


def validate_batch_response(result, expected_keys):
    if not isinstance(result, dict) or not isinstance(result.get("responses"), list):
        raise ValueError("API 응답에 responses 배열이 없습니다.")

    responses = result["responses"]
    actual_keys = [item.get("agent_key") for item in responses]

    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("API 응답의 agent_key가 중복되었습니다.")

    if set(actual_keys) != set(expected_keys):
        raise ValueError(
            f"API 응답 ID 불일치: 예상={expected_keys}, 실제={actual_keys}"
        )

    by_key = {}
    for item in responses:
        clean = {}
        for question in QUESTION_COLS:
            value = item.get(question)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{item.get('agent_key')} / {question}가 정수가 아닙니다.")
            if not 1 <= value <= 7:
                raise ValueError(f"{item.get('agent_key')} / {question} 범위 오류: {value}")
            clean[question] = value
        by_key[item["agent_key"]] = clean

    return by_key


def ask_batch(client, model, system_prompt, items):
    agent_keys = [item["agent_key"] for item in items]
    schema = make_batch_schema(agent_keys)
    user_prompt = make_batch_user_prompt(items)

    request = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
        "temperature": 0.9,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "survey_batch_response",
                "strict": True,
                "schema": schema,
            }
        },
    }

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.responses.create(**request)
            result = json.loads(response.output_text)
            return validate_batch_response(result, agent_keys)
        except Exception as error:
            print(f"[API 오류] 시도 {attempt}/{MAX_RETRY}: {error}")
            if attempt == MAX_RETRY:
                raise
            wait_seconds = min(2 ** attempt, 60)
            print(f"{wait_seconds}초 후 다시 시도합니다.")
            time.sleep(wait_seconds)


# ============================================================
# 5. 결과 요약과 검증
# ============================================================

def summarize_questions(data, group_cols=None):
    group_cols = group_cols or []
    rows = []

    if group_cols:
        grouped = data.groupby(group_cols, dropna=False)
    else:
        grouped = [("전체", data)]

    for group_key, temp in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_info = dict(zip(group_cols, group_key))

        for question in QUESTION_COLS:
            scores = pd.to_numeric(temp[question], errors="coerce")
            valid_n = int(scores.notna().sum())
            row = {
                **group_info,
                "문항": question,
                "N": valid_n,
                "평균": round(scores.mean(), 4),
                "표준편차": round(scores.std(), 4),
                "중앙값": round(scores.median(), 4),
            }

            for score in range(1, 8):
                count = int((scores == score).sum())
                percentage = count / valid_n * 100 if valid_n else 0.0
                row[f"{score}점_건수"] = count
                row[f"{score}점_비율(%)"] = round(percentage, 4)

            rows.append(row)

    return pd.DataFrame(rows)


def collect_validation_errors(original, final, id_col):
    errors = []

    if len(final) != len(original):
        errors.append("입력/출력 행 수 불일치")

    if final[id_col].duplicated().any():
        errors.append(f"{id_col} 중복")

    for question in QUESTION_COLS:
        numeric = pd.to_numeric(final[question], errors="coerce")
        invalid = (
            numeric.isna()
            | (numeric < 1)
            | (numeric > 7)
            | (numeric % 1 != 0)
        )
        if invalid.any():
            errors.append(f"{question} 값 오류 {int(invalid.sum()):,}건")

    return errors


# ============================================================
# 6. API 정보 준비
# ============================================================

def prepare_api_client():
    model = os.getenv("OPENAI_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not model:
        model = input("사용할 OpenAI 모델 ID를 입력하세요: ").strip()
        if not model:
            raise ValueError("모델 ID가 입력되지 않았습니다.")

    if not api_key:
        api_key = getpass("OpenAI API 키를 입력하세요(화면에 표시되지 않음): ").strip()
        if not api_key:
            raise ValueError("API 키가 입력되지 않았습니다.")

    if base_url is None and sys.stdin.isatty():
        base_url = input(
            "별도 API 주소(Base URL)가 있으면 입력하고, 없으면 Enter를 누르세요: "
        ).strip()

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs), model


# ============================================================
# 7. 개별 실험 실행
# ============================================================

def run_experiment(mode, client, model):
    print("\n" + "=" * 64)
    print(f"{mode} 실험 시작")
    print("=" * 64)

    data, id_col, input_cols, unique_input = prepare_input_data(mode)
    system_prompt = A_SYSTEM_PROMPT if mode == "A" else N_SYSTEM_PROMPT
    cache_file = cache_file_for_run(mode)
    cache = load_cache(cache_file)

    answers_by_key = {}
    missing_indices = []

    for index, row in unique_input.iterrows():
        cache_key = make_cache_key(mode, model, row, input_cols)
        if cache_key in cache:
            answers_by_key[cache_key] = cache[cache_key]
        else:
            missing_indices.append(index)

    print(f"새로 요청할 고유 입력 수: {len(missing_indices):,}")

    completed = len(unique_input) - len(missing_indices)
    for start in range(0, len(missing_indices), BATCH_SIZE):
        indices = missing_indices[start:start + BATCH_SIZE]
        batch = unique_input.loc[indices]
        items, key_to_row = build_batch_items(mode, batch)
        batch_answers = ask_batch(client, model, system_prompt, items)

        cache_records = []
        for agent_key, row in key_to_row.items():
            cache_key = make_cache_key(mode, model, row, input_cols)
            answer = batch_answers[agent_key]
            cache_records.append((cache_key, answer))
            cache[cache_key] = answer
            answers_by_key[cache_key] = answer

        append_cache_records(cache_file, cache_records)

        completed += len(batch)
        print(f"[진행] {completed:,} / {len(unique_input):,} 고유 입력")

    result_rows = []
    for _, row in unique_input.iterrows():
        cache_key = make_cache_key(mode, model, row, input_cols)
        result_row = {column: row[column] for column in input_cols}
        result_row.update(answers_by_key[cache_key])
        result_rows.append(result_row)

    answer_df = pd.DataFrame(result_rows)
    final = data.merge(answer_df, on=input_cols, how="left", validate="many_to_one")

    if mode == "A" and H4_SURVEY_CONSISTENCY:
        from h4_calibrate import derive_h4_from_survey

        final = derive_h4_from_survey(final, id_col=id_col)
        print("[H4 추론] 동일 응답자 H2·H3·이동특성 기반 H4 일관화 (실측 복사 없음)")

    elif mode == "A" and H4_SOFT_CONSISTENCY:
        from h4_calibrate import nudge_h4_consistency

        final = nudge_h4_consistency(final, id_col=id_col)
        print("[H4 조정] H2·H3·이동부담 기반 부드러운 일관성 조정(±1)")

    validation_errors = collect_validation_errors(data, final, id_col)

    run_suffix = "test" if TEST_LIMIT is not None else "final"
    output_prefix = f"{mode}_{run_suffix}"

    if mode == "A":
        metadata_cols = [
            "Synthetic_ID",
            "Persona_ID",
            "아키타입ID",
            "아키타입명",
            "프로필구분",
            *FACT_COLS,
        ]
    else:
        metadata_cols = [
            "Nemotron_ID",
            "sex",
            "age",
            "marital_status",
            "family_type",
            "housing_type",
            "education_level",
            "occupation",
            "province",
            "district",
        ]

    response_cols = [column for column in metadata_cols if column in final.columns]
    response_cols += QUESTION_COLS

    response_file = RESULT_DIR / f"{output_prefix}_responses.csv"
    final[response_cols].sort_values(id_col).to_csv(
        response_file,
        index=False,
        encoding="utf-8-sig",
    )

    overall = summarize_questions(final)
    overall_file = RESULT_DIR / f"{output_prefix}_summary_overall.csv"
    overall.to_csv(overall_file, index=False, encoding="utf-8-sig")

    if mode == "A":
        for group_col, file_tag in [
            ("Persona_ID", "페르소나"),
            ("아키타입ID", "아키타입"),
        ]:
            if group_col in final.columns:
                group_summary = summarize_questions(final, [group_col])
                group_summary.to_csv(
                    RESULT_DIR / f"{output_prefix}_summary_{file_tag}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    validation_file = RESULT_DIR / f"{output_prefix}_validation.txt"
    with validation_file.open("w", encoding="utf-8") as handle:
        handle.write(f"MODE: {mode}\n")
        handle.write(f"MODEL: {model}\n")
        handle.write(f"PROMPT_VERSION: {PROMPT_VERSION}\n")
        handle.write(f"TEST_LIMIT: {TEST_LIMIT}\n")
        handle.write(f"입력 행 수: {len(data)}\n")
        handle.write(f"출력 행 수: {len(final)}\n")
        handle.write(f"{id_col} 고유값: {final[id_col].nunique()}\n")
        handle.write(f"고유 API 입력 수: {len(unique_input)}\n")
        if validation_errors:
            handle.write("\n검증 결과: 실패\n")
            for error in validation_errors:
                handle.write(f"- {error}\n")
        else:
            handle.write("\n검증 결과: 통과\n")

    print("\n" + "=" * 64)
    print(f"{mode} 실험 완료")
    print("=" * 64)
    print(f"응답파일: {response_file}")
    print(f"전체 통계: {overall_file}")
    print(f"검증파일: {validation_file}")

    if validation_errors:
        print("\n검증 실패")
        for error in validation_errors:
            print(f"- {error}")
        raise ValueError("응답 검증에 실패했습니다.")

    print("\n검증 통과")
    return response_file


# ============================================================
# 8. A → N → 최종 비교 통합 실행
# ============================================================

def evaluate_partial_h4(response_file: Path) -> None:
    from compare_h4 import (
        ACTUAL_2021_PCT,
        MINIMUM_LEVEL,
        TARGET_LEVEL,
        calculate_metrics,
        level_label,
        make_distribution,
    )

    data = pd.read_csv(response_file, encoding="utf-8-sig")
    scores = pd.to_numeric(data["H4"], errors="coerce")
    invalid = scores.isna() | ~scores.isin(range(1, 8))
    if invalid.any():
        raise ValueError("H4 값이 올바르지 않습니다.")

    _, percentages = make_distribution(scores.astype(int))
    actual = pd.Series(ACTUAL_2021_PCT, dtype=float)
    similarity, mae, tv = calculate_metrics(actual, percentages)

    report_file = RESULT_DIR / "A_partial_H4_evaluation.txt"
    with report_file.open("w", encoding="utf-8") as handle:
        handle.write("[A 부분 실행 H4 평가 — 실측 복사 없음]\n\n")
        handle.write(f"표본: {len(data):,}명\n")
        handle.write(f"PROMPT: {PROMPT_VERSION}\n")
        handle.write(f"분포유사도: {similarity:.4f}%\n")
        handle.write(f"MAE: {mae:.4f}%p\n")
        handle.write(f"TV: {tv:.4f}%p\n")
        handle.write(f"최소 실용({MINIMUM_LEVEL:.0f}%): {level_label(similarity)}\n")
        handle.write(f"목표({TARGET_LEVEL:.0f}%) 대비: {similarity - TARGET_LEVEL:+.2f}%p\n")

    print("\n" + "=" * 64)
    print("부분 표본 H4 유사도 (실측 복사 없음)")
    print("=" * 64)
    print(f"N={len(data):,}  유사도={similarity:.2f}%  판정={level_label(similarity)}")
    print(f"평가파일: {report_file}")


def main():
    validate_settings()
    client, model = prepare_api_client()

    last_response_file = None
    for mode in RUN_MODES:
        last_response_file = run_experiment(mode, client, model)

    if TEST_LIMIT is not None:
        if last_response_file is not None:
            evaluate_partial_h4(last_response_file)
        print("\n" + "=" * 64)
        print("A/N 테스트 실행 완료")
        print("=" * 64)
        print(
            f"현재 TEST_LIMIT={TEST_LIMIT}이므로 최종 H4 비교는 실행하지 않습니다."
        )
        print("10,000명 최종 실행은 TEST_LIMIT = None으로 설정하세요.")
        return

    if RUN_COMPARISON:
        print("\n" + "=" * 64)
        print("A/N 생성 완료 - H4 최종 비교 시작")
        print("=" * 64)

        from compare_h4 import main as compare_h4_main

        compare_h4_main()

    print("\n" + "=" * 64)
    print("모든 작업 완료")
    print("=" * 64)


if __name__ == "__main__":
    main()

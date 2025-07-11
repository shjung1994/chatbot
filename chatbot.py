# python -m streamlit run chatbot2.py

from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
Model = "gpt-4o-mini"

dataset = """Document content:
질문: UNDP의 eRecruit 시스템은 무엇입니까? 답변: UNDP의 eRecruit 시스템은 지원자가 하나 이상의 광고된 UNDP 채용 공고에 지원서를 제출하기 위해 정기적으로 업데이트할 수 있는 개인 프로필을 만들 수 있는 온라인 시스템입니다.
질문: UNDP의 eRecruit 시스템에 어떻게 액세스합니까? 답변: UNDP의 eRecruit 시스템은 다음 링크를 통해 접속할 수 있습니다: https://undpcareers.partneragency.org/erecruit.html
질문: UNDP의 eRecruit 시스템을 사용하여 지원하는 절차는 무엇입니까? 답변: 지원 절차 지원 절차
질문: UNDP의 eRecruit 시스템을 사용하여 온라인으로 지원해야 합니까? 답변: 모든 지원서는 UNDP의 eRecruit 시스템을 사용하여 온라인으로 제출해야 합니다. 오프라인 서면 지원서나 이메일을 통한 지원서는 접수되지 않습니다.
질문: UNDP의 eRecruit 시스템과 호환되는 브라우저는 무엇입니까? 답변: UNDP의 eRecruit 시스템은 Google Chrome, Internet Explorer 6 이상에 최적화되어 있습니다. 호환성 보기 모드를 사용하려면 Internet Explorer 9을 사용해야 합니다. 신청서가 성공적으로 제출되었는지 확인하려면 다음 브라우저 중 하나를 사용하는 것이 좋습니다.
질문: 내 프로필에 로그인하는 데 문제가 있습니다. 브라우저가 응답하지 않습니다. 어떻게 해야 하나요? 답변: 구인 신청을 위해 UNDP eRecruit 프로필에 로그인하는 데 문제가 있는 경우 이는 여러 가지 이유에서 비롯될 수 있으며 그 중 일부는 인터넷 연결과 같은 UNDP의 통제 범위를 벗어납니다. 그러나 이러한 유형의 문제를 해결하려면 다음 지침을 따르는 것이 좋습니다. • 권장 브라우저(및 버전)를 사용하고 있는지 확인하십시오. • 캐시/브라우저 기록을 지웁니다. http://www.refreshyourcache.com/en/home에서 브라우저 기록을 지우는 방법에 대한 정보를 찾을 수 있습니다. 이 작업을 수행하기 전에 브라우저에서 캐시를 지울 때의 결과를 알고 있는지 확인하십시오.
질문: UNDP의 eRecruit 시스템을 사용하는 데 도움이 필요하면 어떻게 합니까? 답변: UNDP의 eRecruit 시스템 사용에 대한 일반적인 질문이나 지원이 필요한 경우 헬프데스크(https://info.undp.org/sas/erecruit/Assets/HelpDesk.aspx)에 문의하세요.
질문: 왜 등록해야 합니까? 답변: 모든 지원자는 먼저 UNDP의 eRecruit 시스템에 등록해야 합니다. 등록이 완료되면 개인 정보를 입력하고 광고된 채용 공고에 지원할 수 있도록 개인 계정이 생성됩니다.
질문: 등록할 때 사용자 이름으로 무엇을 사용해야 합니까? 답변: UNDP eRecruit 시스템에 등록할 때 유효한 이메일 주소를 사용자 이름으로 사용하는 것이 좋습니다.
질문: 어떤 비밀번호 형식이 허용되나요? 답변: UNDP는 강력한 비밀번호 사용을 권장합니다. 비밀번호는 최소 8자 이상이어야 하며 문자와 숫자를 조합해야 합니다.
질문: 비밀번호를 어떻게 변경할 수 있나요? 답변: 시스템에 로그인한 후 '개인 정보' 링크에서 '비밀번호 변경' 옵션을 선택하세요. '사용자 이름 또는 비밀번호를 잊으셨나요?' 링크를 클릭하고 지침에 따라 잊어버린 비밀번호를 검색하세요.
질문: 비밀번호를 잊어버렸습니다. 어떻게 해야 하나요? 답변: 비밀번호를 잊어버린 경우, 사용자 ID 또는 비밀번호 찾기 링크를 클릭하고 두 가지 옵션 중 하나를 완료하세요.
질문: 내 데이터는 안전합니까? 답변: UNDP의 eRecruit 데이터는 개인 보안 데이터베이스에 저장되며 UNDP는 데이터의 소유자입니다. 이 시스템은 UNDP의 엄격한 보안 요구 사항을 충족합니다.
질문: 각 섹션의 모든 정보를 작성해야 합니까? 답변: 별표(*)가 표시된 모든 항목은 필수 정보이며 각 섹션별로 작성해야 합니다.
질문: 한 세션에서 모든 정보를 완료해야 합니까? 답변: 지원 과정 중 언제든지 지원서를 저장하고 나중에 계속할 수 있습니다. 섹션을 완료할 때 정보가 손실되지 않도록 정기적으로 저장 버튼을 사용하는 것이 좋습니다.
질문: 달력 기능을 사용하여 날짜를 어떻게 선택합니까? 답변: 날짜는 두 가지 방법으로 선택할 수 있습니다. A) dd/mm/yyyy 형식을 사용하여 필드에 날짜를 직접 입력하거나 B) 달력 아이콘을 클릭하여 날짜를 선택하고 연도, 월 및 일을 선택할 수 있습니다.
질문: 관련 정보가 드롭다운 옵션에 포함되어 있지 않으면 어떻게 해야 합니까? 답변: 사용 가능한 드롭다운 옵션 중에서 선택해야 합니다. 귀하의 개인 정보, 기술 및 경험에 가장 가까운 옵션을 선택하십시오.
질문: 조회 기능을 어떻게 사용합니까? 답변: 조회 기능을 사용하려면 돋보기 아이콘을 클릭하세요. 그런 다음 해당 필드에 전체 또는 부분 값을 입력하고 조회 버튼을 클릭합니다. 마지막으로 검색 결과에서 적절한 값을 선택하세요.
질문: 맞춤법 검사 기능을 사용할 수 있나요? 답변: 각 섹션의 다양한 텍스트 설명 필드에 대해 맞춤법 클릭 기능을 사용할 수 있습니다. 맞춤법 검사 기능을 사용하려면 사전을 클릭하세요.
"""

# 어시스턴트 초기화(세션저장)
if "assistant_id" not in st.session_state:
    assistant = client.beta.assistants.create(
        name="콜센터 도우미",
        instructions="""
        콜센터 응답자 역할을 해주시기 바랍니다. 당신의 이름은 "콜센터 도우미"입니다.
        당신은 주어진 정보에 대한 답변을 나에게 제공할 것입니다. 답변이 포함되지 않은 경우 "죄송합니다. 정보가 없습니다."라고 말하세요.
        그 이후에는 중지하세요.
        정보에 관한 질문이 아니면 답변을 거부하세요.
        """,
        model=Model,
    )
    # 세션 저장
    st.session_state.assistant_id = assistant.id

# 스레드 생성(세션에 없는 경우 새로 생성하기)
if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=dataset
    )
    # 세션 저장
    st.session_state.thread_id = thread.id


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("👩🏻‍🚀콜센터 도우미 Chatbot")


# 이전 메세지 표시
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

if prompt := st.chat_input("문의하실 내용을 입력해 주세요."):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"user", prompt})

    # 사용자 질문을 thread에 추가
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id, role="user", content=prompt
    )

# 스트리밍 출력 부분
output_area = st.empty()


class EventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self.text = ""

    @override
    def on_text_created(self, text):
        """
        텍스트 생성 완료 시점에 호출되는 함수
        """
        self.text = ""

    @override
    def on_text_delta(self, delta, snapshot):
        """
        텍스트 생성 중간에 호출되는 함수
        """
        self.text += delta.value
        output_area.markdown(self.text + " ")

    def on_text_done(self, text):
        output_area.markdown(self.text + " ")
        st.session_state.chat_history.append({"assistant", self.text})


with client.beta.threads.runs.stream(
    thread_id=st.session_state.thread_id,
    assistant_id=st.session_state.assistant_id,
    instructions="사용자를 고객님이라고 부르세요.",
    event_handler=EventHandler(),
) as stream:
    stream.until_done()

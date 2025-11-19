import asyncio
import json
import logging
from abc import ABC
from enum import Enum
from flask import Flask, render_template
from flask_socketio import SocketIO
import asyncio
from gemini_adapter import GeminiAPIHandler
from google.genai import types
import threading  # added

class Team(Enum):
    PRO = 1
    CON = 2
    JUDGE = 3


# 設置 Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
debater_loggers = {
    Team.PRO: logging.getLogger("ProDebater"),
    Team.CON: logging.getLogger("ConDebater"),
    Team.JUDGE: logging.getLogger("Judge"),
}
# judge_logger = logging.getLogger("Judge")

debate_handlers = {
    Team.PRO: logging.FileHandler(
        "debate_output/debater_pro.log", mode="w", encoding="utf-8"
    ),
    Team.CON: logging.FileHandler(
        "debate_output/debater_con.log", mode="w", encoding="utf-8"
    ),
    Team.JUDGE: logging.FileHandler(
        "debate_output/judge.log", mode="w", encoding="utf-8"
    ),
}
# judge_handler = logging.FileHandler(
#     "debate_output/judge.log", mode="w", encoding="utf-8"
# )

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

for logger, handler in zip(debater_loggers.values(), debate_handlers.values()):
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# judge_handler.setFormatter(formatter)
# judge_logger.addHandler(judge_handler)
app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")

def _convert_messages(messages, default_system="You are a helpful assistant."):
    """Convert list[{'role','content'}] to (contents, system_prompt) for Gemini."""
    system_prompt = default_system
    contents = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_prompt = m["content"]
            continue
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=m["content"])]
            )
        )
    return contents, system_prompt

class Debater(ABC):
    def __init__(self, name: str, topic, api: GeminiAPIHandler):
        self.name = name
        self.topic = topic
        self.team = Team.PRO if name == "正方" else Team.CON
        self.logger = debater_loggers[self.team]
        self.api = api
        self.round_score = 0
        self.total_score = 0
        self.arguments = []
        self.memory = []

    async def prepare_arguments(self, num_args: int = 5):
        messages = [
            {
                "role": "user",
                "content": f'你正在參加一場辯論賽，請提供{num_args}個{("支持" if self.team == Team.PRO else "反對")}「{self.topic}」的論點, 論點為字串，格式使用 python list format, 範例: ["論點1", "論點2", ...]',
            }
        ]
        contents, system_prompt = _convert_messages(messages)
        response = await self.api.generate_content_v1(contents, system_prompt)
        response = response[response.find("[") : response.rfind("]")+1]
        parsed_response = json.loads(response)
        self.arguments.extend(parsed_response)
        for i, arg in enumerate(self.arguments, 1):
            self.logger.info(f"{self.name} 準備論點 {i:2d}: {arg}")

    async def rebut(self, opponent_argument: str, T: float):
        T = round(T, 3)
        messages = [
            {
                "role": "model",
                "content": f'我{("支持" if self.team == Team.PRO else "反對")}「{self.topic}」，因為'
                + ", ".join(self.arguments),
            },
            {
                "role": "user",
                "content": f'我{("反對" if self.team == Team.PRO else "支持")}「{self.topic}」，因為'
                + opponent_argument,
            },
            {
                "role": "user",
                "content": f'你正在參加一場辯論賽，要回饋對手的論點，請依照指示的對抗強度(範圍0~1, 0: 融合對方論點，尋找共識平衡點, 1: 質疑對方可行性與可靠性)回饋，目前對抗強度={T}',
            },
        ]
        contents, system_prompt = _convert_messages(messages)
        response = await self.api.generate_content_v1(contents, system_prompt)
        self.memory.append(response)
        self.logger.info(f"{self.name} (T={T}) 反駁「{opponent_argument}」: {response}")


class Judge:
    """裁判評分系統"""

    def __init__(self, api: GeminiAPIHandler):
        self.api = api

    async def evaluate(self, arg: str):
        """根據可靠度和合理程度計算得分"""
        messages = [
            {
                "role": "user",
                "content": f"你是一位辯論賽評審，請先客觀分析選手應答，且根據你對於應答內容的分析，分別給出兩個整數(範圍0~10)，代表選手應答內容之可靠度和有效反駁程度\n選手回應:{arg}",
            }
        ]
        contents_step1, system_prompt_step1 = _convert_messages(messages, default_system="你是辯論賽評審。")
        response_step1 = await self.api.generate_content_v1(contents_step1, system_prompt_step1)

        jsonPrompt = [
            {
                "role": "system",
                "content": "given a review of a debate response from a judge, please extract the score and analysis from the review. The review is in Chinese. The json should contain the following fields: 'analysis', 'credibility', 'validity', e.g.:\n{\"analysis\": \"string\", \"credibility\": int, \"validity\": int}",
            },
            {
                "role": "user",
                "content": response_step1,
            },
        ]
        contents_step2, system_prompt_step2 = _convert_messages(jsonPrompt)
        response_step2 = await self.api.generate_content_v1(contents_step2, system_prompt_step2)
        response_step2 = response_step2[response_step2.find("```json") + 7 : response_step2.rfind("```")]
        parsed_response = json.loads(response_step2)
            
        return parsed_response["analysis"], (parsed_response["credibility"], parsed_response["validity"])


class DebateController:
    """控制辯論過程"""

    def __init__(self, topic: str, rounds: int, prepare: int):
        self.topic = topic
        self.rounds = rounds
        self.prepare = prepare
        self.resume_event = threading.Event()  # added

    async def wait_for_resume(self):  # added
        socketio.emit("update_judge", {"text": "等待使用者開始下一回合..."})
        socketio.emit("await_resume", {"text": "請點擊『下一回合』繼續"})
        self.resume_event.clear()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.resume_event.wait)

    async def start_debate(self):
        logging.info(f"辯論主題: {self.topic}")
        self.api = GeminiAPIHandler()
        self.pro = Debater("正方", self.topic, self.api)
        self.con = Debater("反方", self.topic, self.api)
        self.judge = Judge(self.api)
        socketio.emit(
            "update_pro", {"text": "正方準備論點中... (請稍候)"}
        )
        socketio.emit(
            "update_con", {"text": "反方準備論點中... (請稍候)"}
        )
        await self.pro.prepare_arguments(self.prepare)
        await self.con.prepare_arguments(self.prepare)
        for arg in self.pro.arguments:
            socketio.emit("update_pro", {"text": arg})
        for arg in self.con.arguments:
            socketio.emit("update_con", {"text": arg})
        # 新增：讓使用者檢視生成的論點後再開始辯論
        socketio.emit("update_judge", {"text": "檢視雙方論點後按『下一回合』開始辯論"})
        await self.wait_for_resume()
        T_start = 0.9
        T_delta = 2
        for i in range(self.rounds):
            T = T_start / T_delta**i
            logging.info(f"回合 {i+1}/{self.rounds}, T={T}")
            socketio.emit(
                "update_judge", {"text": f"回合 {i+1}/{self.rounds}, T={T}"}
            )
            socketio.emit(
                "update_pro", {"text": f"回合 {i+1}/{self.rounds}"}
            )
            socketio.emit(
                "update_con", {"text": f"回合 {i+1}/{self.rounds}"}
            )
            for j, arg in enumerate(self.con.arguments):
                await self.pro.rebut(arg, T)
                socketio.emit(
                    "update_pro",
                    {"text": f"論點 {j+1}/{self.prepare}\n{self.pro.memory[-1]}"},
                )
            for j, arg in enumerate(self.pro.arguments):
                await self.con.rebut(arg, T)
                socketio.emit(
                    "update_con",
                    {"text": f"論點 {j+1}/{self.prepare}\n{self.con.memory[-1]}"},
                )

            for team in [self.pro, self.con]:
                scores = [0.0] * len(team.memory)
                for i_mem, rebut in enumerate(team.memory):  # renamed i to i_mem to avoid shadowing
                    analysis, (credibility, validity) = await self.judge.evaluate(rebut)
                    scores[i_mem] = (credibility, validity, credibility * validity)
                    socketio.emit(
                        "update_judge", {"text": f"{team.name} {i_mem+1}/{len(team.memory)}: {credibility}, {validity}, {analysis}"}
                    )
                team.round_score = sum(score[2] for score in scores)
                team.memory = []

            if self.pro.round_score > self.con.round_score:
                self.pro.total_score += 1
                socketio.emit(
                    "update_judge", {"text": f"正方: {self.pro.round_score}, 反方: {self.con.round_score}, 正方勝"}
                )
            elif self.pro.round_score < self.con.round_score:
                self.con.total_score += 1
                socketio.emit(
                    "update_judge", {"text": f"正方: {self.pro.round_score}, 反方: {self.con.round_score}, 反方勝"}
                )
            else:
                socketio.emit(
                    "update_judge", {"text": f"正方: {self.pro.round_score}, 反方: {self.con.round_score}, 平手"}
                )

            # wait for user to resume unless last round
            if i < self.rounds - 1:
                await self.wait_for_resume()

        if self.pro.total_score > self.con.total_score:
            result = "正方勝"
        elif self.pro.total_score < self.con.total_score:
            result = "反方勝"
        else:
            result = "平手"
        socketio.emit(
            "update_judge", {"text": f"最終結果: {result}"}
        )
        logging.info(f"最終結果: {result}")


@app.route("/")
def index():
    return render_template("index.html")

current_debate = None  # added

@socketio.on('start_debate')
def handle_start_debate(data):
    global current_debate
    topic = data['topic']
    rounds = data['rounds']
    prepare_amount = data['prepare_amount']
    current_debate = DebateController(topic, rounds, prepare_amount)
    asyncio.run(current_debate.start_debate())

@socketio.on('resume_debate')  # added
def handle_resume_debate():
    global current_debate
    if current_debate is not None:
        current_debate.resume_event.set()

# 測試
if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
    # topic = "台灣政府應在 2025 年前淘汰所有核電廠"
    # topic = "遊戲中上不去分完全歸因於玩家自身"
    # debate = DebateController(topic, rounds=3, prepare=2)
    # asyncio.run(debate.start_debate())
    # print(f"勝者: {result}")

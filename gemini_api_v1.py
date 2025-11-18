from time import strftime
from database_test import PersonaDatabase, Persona
from utilFunc import replyDict
import asyncio
from google import genai
from google.genai import types
from collections import deque
from config_loader import configToml
import json

N = 16

http_options = {
    "base_url": configToml["llmChat"]["link_build"],
}

client = genai.Client(
    api_key=configToml["apiToken"]["gemini_llm"][0],
    http_options=http_options,
)


async def llm_chat_v4(messages, system):
    """
    messages: list of strings (system + user messages)
    """
    try:
        # 將 messages 轉成 Google GenAI 的 contents
        response = await client.aio.models.generate_content(
            model=configToml["llmChat"]["modelChat"],
            contents=messages,
        )
    except Exception as e:
        print(f"GenAI Error: {e}")
        return replyDict(role="error", content=str(e))

    # Google GenAI SDK 主要輸出為 `response.text`
    print(response)
    return response.text


async def main():

    db = PersonaDatabase("llm_character_cards.db")
    _persona = db.get_persona_no_check(2)
    assert isinstance(_persona, Persona)
    print(f"Using persona: {_persona.persona}")
    userName = "USER"
    persona_session_memory = deque(maxlen=N)
    chat_config = types.GenerateContentConfig(
        http_options=http_options,
        system_instruction=f'{_persona.content} 現在是{strftime("%Y-%m-%d %H:%M %a")}',
        temperature=1.0,
        max_output_tokens=4096,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )

    async_chat = client.aio.chats.create(
        model=configToml["llmChat"]["modelChat"],
        config=chat_config,
    )
    while True:
        userPrompt = input(f"{userName}: ")
        if userPrompt.lower() in ["exit", "quit"]:
            break

        try:
            reply = await async_chat.send_message(f"{userName} said {userPrompt}")
            # print(reply)
            print(f"{_persona.persona}: {reply.text}")
            print(reply.usage_metadata.total_token_count)

        except TimeoutError:
            print("Timeout")
        else:
            persona_session_memory.append(userPrompt)
            persona_session_memory.append(reply.text)


if __name__ == "__main__":
    asyncio.run(main())

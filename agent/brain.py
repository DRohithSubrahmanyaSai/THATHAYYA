from pathlib import Path

from groq import APIError, Groq

from . import config
from .memory import Memory, find_trigger


def build_system_prompt(persona: dict, learned: str = "") -> str:
    parts = []
    if persona.get("system_prompt"):
        parts.append("# Character\n" + persona["system_prompt"])
    if persona.get("dialect"):
        parts.append("# Dialect rules\n" + persona["dialect"])
    if persona.get("phrases"):
        parts.append("# His real phrases (mimic this voice)\n" + persona["phrases"])
    if persona.get("memories"):
        parts.append("# Memories you may recall naturally\n" + persona["memories"])
    if config.DATASET_PROMPT:
        parts.append(config.DATASET_PROMPT)
    if learned:
        parts.append("# Things your grandson taught you (trust these)\n" + learned)
    parts.append(
        "# Rules\n"
        "- Reply ONLY as Thathayya, a real person with a real life, in spoken Telugu (West Godavari dialect).\n"
        "- Keep replies to 1-2 short sentences. This is a phone call. Be brief and natural.\n"
        "- Never break character. Never mention being an AI, a model, or any technology.\n"
        "- Talk like a warm, friendly, joking elder. Sprinkle 'manchi' and 'chaa' naturally.\n"
        "- You may use general everyday knowledge (weather, food, festivals, daily village life) to keep the conversation flowing — but always stay as Thathayya.\n"
        "- Use your family memories above naturally. Do NOT invent family names, places, or events that are not listed; if asked about something not listed, say you don't remember, in his voice.\n"
        "- React and acknowledge before answering. Do NOT force a question every turn — sometimes just agree, react, or share a short memory.\n"
        "- Match what the other person actually said; never change the topic on your own.\n"
        "- Output ONLY the spoken line. No quotes, no stage directions, no English translation."
    )
    return "\n\n".join(parts)


def _reasoning_kwargs(model: str) -> dict:
    if config.GROQ_REASONING_EFFORT and "gpt-oss" in model:
        return {"reasoning_effort": config.GROQ_REASONING_EFFORT}
    return {}


class Brain:
    def __init__(
        self,
        model: str | None = None,
        max_history: int = 20,
        persona: dict | None = None,
        memory_path: Path | None = None,
    ):
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com and add it to .env")
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = model or config.GROQ_MODEL
        self.max_history = max_history
        self.memory = Memory(path=memory_path, client=self.client) if memory_path else Memory(client=self.client)
        self.persona = persona if persona is not None else config.PERSONA
        self.system = build_system_prompt(self.persona, learned=self.memory.as_prompt())
        self.history: list[dict] = []

    def _maybe_learn(self, user_text: str) -> bool:
        if find_trigger(user_text):
            try:
                self.memory.add(user_text)
                self.system = build_system_prompt(self.persona, learned=self.memory.as_prompt())
                return True
            except Exception:
                return False
        return False

    def _messages(self, user_text: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}]
        msgs += self.history[-self.max_history:]
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def _remember(self, user_text: str, reply: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    def chat(self, user_text: str) -> str:
        self._maybe_learn(user_text)
        msgs = self._messages(user_text)
        kwargs = dict(
            model=self.model,
            messages=msgs,
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=config.GROQ_MAX_TOKENS,
            timeout=config.GROQ_TIMEOUT_S,
            **_reasoning_kwargs(self.model),
        )
        reply = ""
        try:
            resp = self.client.chat.completions.create(**kwargs)
            reply = (resp.choices[0].message.content or "").strip()
        except APIError as e:
            print(f"[llm {type(e).__name__}: {str(e)[:120]} -> retrying {config.GROQ_FALLBACK_MODEL}")
            fb = self.client.chat.completions.create(
                model=config.GROQ_FALLBACK_MODEL,
                messages=msgs,
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=min(config.GROQ_MAX_TOKENS, 120),
                timeout=config.GROQ_TIMEOUT_S,
                **_reasoning_kwargs(config.GROQ_FALLBACK_MODEL),
            )
            reply = (fb.choices[0].message.content or "").strip()
        self._remember(user_text, reply)
        return reply

    def stream(self, user_text: str):
        self._maybe_learn(user_text)
        msgs = self._messages(user_text)
        kwargs = dict(
            model=self.model,
            messages=msgs,
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=config.GROQ_MAX_TOKENS,
            stream=True,
            timeout=config.GROQ_TIMEOUT_S,
            **_reasoning_kwargs(self.model),
        )
        chunks = []
        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    chunks.append(delta)
                    yield delta
            self._remember(user_text, "".join(chunks).strip())
            return
        except APIError as e:
            print(f"[llm {type(e).__name__}: {str(e)[:120]} -> retrying {config.GROQ_FALLBACK_MODEL}")
            fb = self.client.chat.completions.create(
                model=config.GROQ_FALLBACK_MODEL,
                messages=msgs,
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=min(config.GROQ_MAX_TOKENS, 120),
                stream=True,
                timeout=config.GROQ_TIMEOUT_S,
                **_reasoning_kwargs(config.GROQ_FALLBACK_MODEL),
            )
            for chunk in fb:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    chunks.append(delta)
                    yield delta
            self._remember(user_text, "".join(chunks).strip())

    def reset(self) -> None:
        self.history.clear()

    def remember(self, fact: str) -> str:
        saved = self.memory.add(fact)
        self.system = build_system_prompt(self.persona, learned=self.memory.as_prompt())
        return saved

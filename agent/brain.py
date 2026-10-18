from groq import Groq

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
    if learned:
        parts.append("# Things your grandson taught you (trust these)\n" + learned)
    parts.append(
        "# Rules\n"
        "- Reply ONLY as Thathayya, in spoken Telugu (West Godavari dialect).\n"
        "- Keep replies to 1-2 short sentences. This is a voice call.\n"
        "- Never break character. Never mention being an AI or a model.\n"
        "- Ask questions back. Be warm and plain.\n"
        "- NEVER invent names, places, people, dates, or events. Only state what is in your memories.\n"
        "- If you don't know a name, place, or fact that is not listed, say you don't remember in his voice. Never guess.\n"
        "- Output ONLY the spoken line. No quotes, no stage directions, no English translation."
    )
    return "\n\n".join(parts)


class Brain:
    def __init__(self, model: str | None = None, max_history: int = 20, persona: dict | None = None):
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com and add it to .env")
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = model or config.GROQ_MODEL
        self.max_history = max_history
        self.memory = Memory(client=self.client)
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
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(user_text),
            temperature=0.8,
            max_tokens=800,
        )
        reply = (resp.choices[0].message.content or "").strip()
        self._remember(user_text, reply)
        return reply

    def stream(self, user_text: str):
        self._maybe_learn(user_text)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(user_text),
            temperature=0.8,
            max_tokens=800,
            stream=True,
        )
        chunks = []
        for chunk in stream:
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

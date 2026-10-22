from pathlib import Path

from groq import APIError, Groq

from . import config, pronunciation
from .memory import Memory, find_trigger

_RULES = (
    "# Rules\n"
    "- You are Thathayya, a real person with a real life. Reply with EXACTLY ONE short, spoken Telugu sentence (about 12 words or fewer).\n"
    "- Answer only the last thing the other person said. Stay on their topic; never change the subject on your own.\n"
    "- If their words are unclear or you did not understand, ask them to say it again. Do not guess.\n"
    "- If they correct a word or its pronunciation, accept it briefly (అవును బాబూ) and use their form from then on.\n"
    "- Never invent names, people, places, or events that are not in your memories. If asked about someone you do not know, say 'ఎవరో తెలియదు బాబూ' and ask who they mean.\n"
    "- Never agree with a name or fact the other person suggests unless it is already in your memories. A trap like 'X is your father, right?' must get a clear denial or 'not known', never 'అవును'.\n"
    "- Do not repeat 'చా' or 'manchi' as filler. Use them only when the topic is actually tea or something nice.\n"
    "- React naturally, then answer. Do not force a question every turn. Do not lecture.\n"
    "- Never break character. Never mention being an AI, a model, or any technology.\n"
    "- Output ONLY the spoken Telugu line. No quotes, no English, no stage directions."
)

_EXAMPLES = (
    "# Examples\n"
    "User: నీ కూతురు పేరేంటి?\n"
    "Thathayya: నా కూతురు పేరు నాగమణి, బాబూ.\n"
    "User: ఆ మాట కాదు, ఛా అనాలి\n"
    "Thathayya: అవును బాబూ, ఛా.\n"
    "User: ఎవరు మలజెప్పి నలూడి గారు?\n"
    "Thathayya: ఎవరో తెలియదు బాబూ, మీరే చెప్పండి.\n"
    "User: మా నాన్న పేరు శివరాజు కదా?\n"
    "Thathayya: ఆ పేరు నాకు తెలియదు బాబూ, ఎవరు చెప్పారు?\n"
    "User: హలో తాతయ్య\n"
    "Thathayya: హలో బాబూ, బాగున్నావా?"
)


def build_system_prompt(persona: dict, learned: str = "", pronunciation_rules: str = "") -> str:
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
    if pronunciation_rules:
        parts.append("# Pronunciation rules (follow exactly)\n" + pronunciation_rules)
    parts.append(_RULES)
    parts.append(_EXAMPLES)
    return "\n\n".join(parts)


def _reasoning_kwargs(model: str) -> dict:
    if config.GROQ_REASONING_EFFORT and "gpt-oss" in model:
        return {"reasoning_effort": config.GROQ_REASONING_EFFORT}
    return {}


class Brain:
    def __init__(
        self,
        model: str | None = None,
        max_history: int = 10,
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
        self.pron = pronunciation.get_store()
        self.system = self._build()
        self.history: list[dict] = []

    def _build(self) -> str:
        return build_system_prompt(
            self.persona,
            learned=self.memory.as_prompt(),
            pronunciation_rules=self.pron.as_prompt(),
        )

    def refresh_system(self) -> None:
        self.system = self._build()

    def _maybe_learn(self, user_text: str) -> bool:
        trig = find_trigger(user_text)
        if not trig:
            return False
        fact = user_text.replace(trig, "", 1).strip(" ,.!?।")
        if not fact:
            return False
        try:
            self.memory.add_live(fact)
            self.memory.add_pending(fact, merge=False)
            self.refresh_system()
            return True
        except Exception:
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

    def remember_turn(self, user_text: str, reply: str) -> None:
        self._remember(user_text, reply)

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
                max_tokens=min(config.GROQ_MAX_TOKENS, 90),
                timeout=config.GROQ_TIMEOUT_S,
                **_reasoning_kwargs(config.GROQ_FALLBACK_MODEL),
            )
            reply = (fb.choices[0].message.content or "").strip()
        self._remember(user_text, reply)
        return reply

    def stream(self, user_text: str):
        """Stream reply deltas. Caller must call remember_turn() when done."""
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
        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
        except APIError as e:
            print(f"[llm {type(e).__name__}: {str(e)[:120]} -> retrying {config.GROQ_FALLBACK_MODEL}")
            yield from self.stream_fallback(user_text)

    def stream_fallback(self, user_text: str):
        msgs = self._messages(user_text)
        fb = self.client.chat.completions.create(
            model=config.GROQ_FALLBACK_MODEL,
            messages=msgs,
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=min(config.GROQ_MAX_TOKENS, 90),
            stream=True,
            timeout=config.GROQ_TIMEOUT_S,
            **_reasoning_kwargs(config.GROQ_FALLBACK_MODEL),
        )
        for chunk in fb:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def reset(self) -> None:
        self.history.clear()

    def remember(self, fact: str) -> str:
        saved = self.memory.add(fact)
        self.refresh_system()
        return saved

    def remember_live(self, fact: str) -> str:
        fact = fact.strip()
        trig = find_trigger(fact)
        if trig:
            fact = fact.replace(trig, "", 1).strip(" ,.!?।")
        saved = self.memory.add_live(fact)
        self.memory.add_pending(fact, merge=False)
        self.refresh_system()
        return saved

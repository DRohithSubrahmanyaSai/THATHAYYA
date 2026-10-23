"""Real-time pronunciation learning.

The client corrects a word during the call (e.g. "చా కాదు చానా"). We:
1. extract the (wrong -> right) pair,
2. rewrite future LLM output before it reaches TTS,
3. add it to the LLM prompt as a pronunciation rule,
4. sync it to a Cartesia pronunciation dictionary,
5. remember the corrected spelling as an STT keyterm for the next call,
6. optionally keep a short WAV reference of the client saying it.
"""

import json
import re
import threading
import time
import wave
from pathlib import Path

import requests

from . import config

_API_BASE = "https://api.cartesia.ai"
_DICT_NAME = "Thathayya - client pronunciations"
_ITEM_LIMIT = 200
_SYNC_LOCK = threading.Lock()

_PROTECTED = {
    "చాలా", "చాలు", "అది", "ఇది", "అవును", "కాదు", "మీరు", "నేను",
    "మనం", "ఇప్పుడు", "టైం", "ఏమి", "ఎలా", "ఎందుకు", "మంచి", "బాబూ",
}

_PAIR_PATTERNS = (
    re.compile(
        r"^\s*([^\s,.!?]{1,25})\s+(?:కాదు|కాదండి)\s+([^\s,.!?]{1,25})"
        r"(?:\s+(?:అనాలి|అనండి|అని|అంటారు|చెప్పండి|చెప్పు))?\s*[.!?]?\s*$"
    ),
    re.compile(
        r"^\s*([^\s,.!?]{1,25})\s+అనకూడదు[,\s]+([^\s,.!?]{1,25})"
        r"(?:\s+(?:అనాలి|అనండి|అని|అంటారు|చెప్పండి|చెప్పు))?\s*[.!?]?\s*$"
    ),
)

_JSON_RE = re.compile(r"\{.*\}", re.S)


def _clean(token: str) -> str:
    return token.strip().strip("\"'“”‘’.,!?;:()[]{}")


def _valid_pair(wrong: str, right: str) -> bool:
    wrong, right = _clean(wrong), _clean(right)
    if not wrong or not right:
        return False
    if wrong == right:
        return False
    if len(wrong) < 2 or len(right) < 1:
        return False
    if len(wrong) > 30 or len(right) > 30:
        return False
    if wrong in _PROTECTED or right in _PROTECTED:
        return False
    return True


def extract_pair_regex(text: str) -> tuple[str, str] | None:
    for pat in _PAIR_PATTERNS:
        m = pat.match(text.strip())
        if not m:
            continue
        wrong, right = _clean(m.group(1)), _clean(m.group(2))
        if _valid_pair(wrong, right):
            return wrong, right
    return None


class PronunciationStore:
    def __init__(self, path: Path | None = None, client=None):
        self.path = Path(path) if path else config.PRON_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._lock = threading.Lock()
        self._data = self._load()
        if not self._data.get("dictionary_id") and config.CARTESIA_PRON_DICT_ID:
            self._data["dictionary_id"] = config.CARTESIA_PRON_DICT_ID
            self._save()

    # --- state ---
    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("items", [])
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return {"version": 1, "dictionary_id": None, "items": []}

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def items(self) -> list[dict]:
        with self._lock:
            return [dict(i) for i in self._data.get("items", [])]

    def count(self) -> int:
        return len(self._data.get("items", []))

    @property
    def dictionary_id(self) -> str | None:
        return self._data.get("dictionary_id")

    def resolve_dictionary_id(self) -> str | None:
        """Adopt the existing Cartesia dictionary by name (read-only lookup)."""
        dict_id = self.dictionary_id
        if dict_id:
            return dict_id
        if not config.CARTESIA_API_KEY or not config.PRON_DICT_ENABLED:
            return None
        headers = {
            "X-API-Key": config.CARTESIA_API_KEY,
            "Cartesia-Version": config.CARTESIA_VERSION,
        }
        try:
            resp = requests.get(f"{_API_BASE}/pronunciation-dicts", headers=headers, timeout=10)
            resp.raise_for_status()
            body = resp.json() or {}
            data = body.get("data") or body.get("dictionaries") or []
            for item in data:
                if (item.get("name") or "").strip() != config.CARTESIA_PRON_DICT_NAME:
                    continue
                dict_id = item.get("id")
                if not dict_id:
                    continue
                with self._lock:
                    self._data["dictionary_id"] = dict_id
                    self._save()
                print(f"[pron dict] using existing '{config.CARTESIA_PRON_DICT_NAME}' -> {dict_id}")
                return dict_id
        except Exception as e:  # noqa: BLE001
            print(f"[pron dict] lookup failed: {str(e)[:140]}")
        return None

    def as_prompt(self) -> str:
        items = self.items()
        if not items:
            return ""
        return "\n".join(
            f'- Write "{i["pronunciation"]}" instead of "{i["text"]}" when speaking.'
            for i in items
        )

    # --- output rewrite ---
    def apply(self, text: str) -> str:
        if not text:
            return text
        items = sorted(self.items(), key=lambda i: len(i.get("text", "")), reverse=True)
        for item in items:
            wrong = item.get("text") or ""
            right = item.get("pronunciation") or ""
            if not wrong or not right:
                continue
            pattern = re.compile(
                r"(?<![\u0C00-\u0C7F])" + re.escape(wrong) + r"(?![\u0C00-\u0C7F])"
            )
            text = pattern.sub(lambda _m, r=right: r, text)
        return text

    # --- learning ---
    def add(self, wrong: str, right: str, source: str = "user correction", ref: str | None = None) -> bool:
        wrong, right = _clean(wrong), _clean(right)
        if not _valid_pair(wrong, right):
            return False
        with self._lock:
            items = self._data.setdefault("items", [])
            existing = next((i for i in items if i.get("text") == wrong), None)
            if existing:
                existing["pronunciation"] = right
                existing["source"] = source
                existing["ts"] = round(time.time(), 3)
                if ref:
                    existing["ref"] = ref
            else:
                items.append(
                    {
                        "text": wrong,
                        "pronunciation": right,
                        "case_sensitive": False,
                        "source": source,
                        "ts": round(time.time(), 3),
                        **({"ref": ref} if ref else {}),
                    }
                )
            if len(items) > _ITEM_LIMIT:
                del items[:-_ITEM_LIMIT]
            self._save()
        self._append_keyterm(right)
        if config.PRON_DICT_ENABLED:
            threading.Thread(target=self._sync, daemon=True).start()
        return True

    def remove(self, wrong: str) -> bool:
        wrong = _clean(wrong)
        with self._lock:
            items = self._data.setdefault("items", [])
            remaining = [i for i in items if i.get("text") != wrong]
            if len(remaining) == len(items):
                return False
            self._data["items"] = remaining
            self._save()
        if config.PRON_DICT_ENABLED:
            threading.Thread(target=self._sync, daemon=True).start()
        return True

    def extract(self, user_text: str, client=None) -> tuple[str, str] | None:
        pair = extract_pair_regex(user_text)
        if pair:
            return pair
        client = client or self._client
        if client is None:
            return None
        try:
            prompt = (
                "You extract pronunciation corrections from a Telugu conversation line.\n"
                "The speaker is telling Thathayya how a word should be said.\n"
                "Return ONLY compact JSON: {\"wrong\":\"...\",\"right\":\"...\"} or the word NONE.\n"
                "Rules:\n"
                "- wrong = the word/phrase the speaker says NOT to use.\n"
                "- right = the word/phrase the speaker wants used instead (Telugu script).\n"
                "- Ignore ordinary disagreement, facts, or small talk.\n"
                "- Never use common words (చాలా, చాలు, అది, ఇది, అవును, కాదు, టైం) as \"wrong\".\n"
                "- If unsure, return NONE.\n"
                f'User line: "{user_text}"'
            )
            resp = client.chat.completions.create(
                model=config.GROQ_LEARN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                timeout=config.GROQ_TIMEOUT_S,
                reasoning_effort="low",
            )
            out = (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            print(f"[pron extract error] {e}")
            return None
        if not out or out.upper() == "NONE":
            return None
        m = _JSON_RE.search(out)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        wrong, right = _clean(str(data.get("wrong", ""))), _clean(str(data.get("right", "")))
        if _valid_pair(wrong, right):
            return wrong, right
        return None

    def _append_keyterm(self, term: str) -> None:
        try:
            path = config.KEYTERMS_LEARNED_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = set()
            if path.exists():
                existing = {
                    ln.strip()
                    for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")
                }
            if term not in existing:
                with path.open("a", encoding="utf-8") as f:
                    f.write(term + "\n")
        except OSError:
            pass

    def save_ref(self, pcm: bytes, sample_rate: int, wrong: str, right: str) -> str | None:
        if not pcm:
            return None
        try:
            config.PRON_REFS_DIR.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^\w\u0C00-\u0C7F-]+", "_", f"{wrong}_{right}")[:40]
            path = config.PRON_REFS_DIR / f"{int(time.time())}_{safe}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(pcm)
            return str(path)
        except (OSError, wave.Error):
            return None

    # --- Cartesia sync ---
    def _payload_items(self) -> list[dict]:
        return [
            {
                "text": i["text"],
                "pronunciation": i["pronunciation"],
                "case_sensitive": bool(i.get("case_sensitive", False)),
            }
            for i in self.items()
        ]

    def _sync(self) -> None:
        if not config.CARTESIA_API_KEY or not config.PRON_DICT_ENABLED:
            return
        headers = {
            "X-API-Key": config.CARTESIA_API_KEY,
            "Cartesia-Version": config.CARTESIA_VERSION,
            "Content-Type": "application/json",
        }
        items = self._payload_items()
        if not items:
            return
        with _SYNC_LOCK:
            dict_id = self.dictionary_id or self.resolve_dictionary_id()
            try:
                if dict_id:
                    merged = self._merge_with_remote(dict_id, items, headers)
                    if merged is None:
                        return
                    resp = requests.patch(
                        f"{_API_BASE}/pronunciation-dicts/{dict_id}",
                        headers=headers,
                        json={"name": config.CARTESIA_PRON_DICT_NAME, "items": merged},
                        timeout=10,
                    )
                    if resp.status_code == 404:
                        dict_id = None
                    else:
                        resp.raise_for_status()
                        print(f"[pron dict] synced {len(merged)} item(s) -> {dict_id}")
                        return
                resp = requests.post(
                    f"{_API_BASE}/pronunciation-dicts/",
                    headers=headers,
                    json={
                        "name": config.CARTESIA_PRON_DICT_NAME,
                        "description": "Words the client taught Thathayya to pronounce.",
                        "items": items,
                        "access": "private",
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                dict_id = (resp.json() or {}).get("id")
                if dict_id:
                    with self._lock:
                        self._data["dictionary_id"] = dict_id
                        self._save()
                    print(f"[pron dict] created {len(items)} item(s) -> {dict_id}")
            except Exception as e:  # noqa: BLE001
                print(f"[pron dict] sync failed: {str(e)[:140]}")

    def _merge_with_remote(self, dict_id: str, local: list[dict], headers: dict) -> list[dict] | None:
        """Preserve the existing dialect entries when adding client corrections."""
        try:
            resp = requests.get(
                f"{_API_BASE}/pronunciation-dicts/{dict_id}", headers=headers, timeout=10
            )
            if resp.status_code == 404:
                return local
            resp.raise_for_status()
            remote = (resp.json() or {}).get("items") or []
        except Exception as e:  # noqa: BLE001
            print(f"[pron dict] merge read failed, not overwriting: {str(e)[:120]}")
            return None
        by_text = {
            str(i.get("text", "")): {
                "text": i.get("text", ""),
                "pronunciation": i.get("pronunciation", ""),
                "case_sensitive": bool(i.get("case_sensitive", False)),
            }
            for i in remote
            if i.get("text") and i.get("pronunciation")
        }
        for item in local:
            by_text[item["text"]] = item
        return list(by_text.values())


_STORE: PronunciationStore | None = None


def get_store() -> PronunciationStore:
    global _STORE
    if _STORE is None:
        _STORE = PronunciationStore()
    return _STORE

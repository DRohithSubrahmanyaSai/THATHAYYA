_END = ".?!।॥\n"


class SentenceStream:
    """Buffer streamed LLM deltas and emit complete sentences.

    Emits on sentence-ending punctuation once a minimum length is reached, and
    hard-splits on spaces if the buffer grows past max_len without punctuation.
    """

    def __init__(self, min_len: int = 8, max_len: int = 160):
        self._buf = ""
        self.min_len = min_len
        self.max_len = max_len

    def push(self, delta: str) -> list[str]:
        self._buf += delta or ""
        out: list[str] = []
        while True:
            idx = self._boundary()
            if idx is None:
                break
            sentence = self._buf[: idx + 1].strip()
            self._buf = self._buf[idx + 1 :]
            if sentence:
                out.append(sentence)
        while len(self._buf) >= self.max_len:
            cut = self._buf.rfind(" ", 0, self.max_len)
            if cut <= 0:
                cut = self.max_len
            piece = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if piece:
                out.append(piece)
        return out

    def _boundary(self):
        for i, ch in enumerate(self._buf):
            if ch in _END and i + 1 >= self.min_len:
                return i
        return None

    def flush(self) -> list[str]:
        sentence = self._buf.strip()
        self._buf = ""
        return [sentence] if sentence else []

"""Byte-level synthetic passkey/needle-retrieval task generator.

No downloads, no tokenizer (raw bytes, vocab=256). Filler text is built fresh on every
call from a small hand-written template/word list via a seeded numpy RNG, so there is no
fixed corpus to memorize across batches (see DECISIONS.md D5).

Terminology used throughout this module and the eval grid:
  - `total_len` / "length" = the length of the filler+passkey "haystack" portion ONLY,
    i.e. BEFORE the query prefix and answer digits are appended. This matches the
    spec's "Training lengths: a mix sampled from 256 to 1024 tokens" / eval grid
    "length in {256, 512, ...}" framing of context length.
  - `depth` = fractional position (0..1) of the passkey sentence within that haystack.
  - The full fed-to-model sequence is always haystack + QUERY_PREFIX + answer digits.

Design note (see DECISIONS.md D7): the MANDATORY in-window sanity check is built so that
the *entire* sequence (haystack + query + answer) fits within `window` tokens, not just
the haystack. That is the only way "near-100% accuracy when the needle is inside the
window" is a meaningful claim about a sliding-window attention model answering from its
last token: if the query+answer tail alone already pushed the sequence past `window`,
a model could fail even with a perfectly-placed needle, for a reason unrelated to memory.
"""
from __future__ import annotations

import numpy as np
import torch

QUERY_PREFIX = "what is the passkey? "

FILLER_TEMPLATES = [
    "the {w} was sitting quietly near the old door.",
    "every morning the {w} would wander through the quiet garden.",
    "nobody remembered why the {w} had been left on the table.",
    "a soft breeze moved through the trees above the {w}.",
    "the report mentioned a {w} but gave no further detail.",
    "somewhere in the distance a {w} made a faint sound.",
    "she wrote a short note about the {w} and put it away.",
    "the old map showed a {w} near the edge of the forest.",
    "he could not recall where he had placed the {w}.",
    "the museum kept a small {w} in a glass case.",
    "it rained for most of the afternoon near the {w}.",
    "the committee discussed the {w} for nearly an hour.",
    "a child pointed at the {w} and laughed quietly.",
    "the engineer measured the {w} twice before writing it down.",
    "there was a faint smell of {w} coming from the kitchen.",
    "the letter described a journey past the distant {w}.",
    "workers carried the {w} across the narrow bridge.",
    "the teacher used a {w} to explain the lesson.",
    "a stray cat curled up beside the warm {w}.",
    "the captain noted the {w} in the ship's log.",
    "clouds gathered slowly above the quiet {w}.",
    "the recipe called for a small amount of {w}.",
    "visitors often stopped to look at the {w}.",
    "the mechanic replaced the worn {w} without comment.",
]

FILLER_WORDS = [
    "lamp", "garden", "river", "box", "chair", "clock", "letter", "bridge", "forest",
    "engine", "candle", "window", "ladder", "basket", "mirror", "statue", "harbor",
    "orchard", "lantern", "hallway", "notebook", "fountain", "cabinet", "meadow",
    "compass", "tunnel", "blanket", "kettle", "shelf", "anchor", "whistle", "satchel",
]


def make_filler(num_chars: int, rng: np.random.Generator) -> str:
    """Freshly-randomized filler text, at least num_chars long, then trimmed exactly."""
    if num_chars <= 0:
        return ""
    parts = []
    total = 0
    while total < num_chars:
        tmpl = rng.choice(FILLER_TEMPLATES)
        if "{w}" in tmpl:
            tmpl = tmpl.format(w=rng.choice(FILLER_WORDS))
        sent = tmpl + " "
        parts.append(sent)
        total += len(sent)
    text = "".join(parts)
    if len(text) < num_chars:
        text = text + " " * (num_chars - len(text))
    return text[:num_chars]


def make_passkey_sentence(rng: np.random.Generator, num_digits: int = 5) -> tuple[str, str]:
    digits = "".join(str(int(d)) for d in rng.integers(0, 10, size=num_digits))
    sentence = f"the passkey is {digits}."
    return sentence, digits


def build_example(total_len: int, depth: float, rng: np.random.Generator, num_digits: int = 5) -> dict:
    """Builds one example's raw bytes as a numpy int64 array of byte values.

    Returns a dict with:
      input_ids   : np.ndarray[int64], shape (total_len + len(QUERY_PREFIX) + num_digits,)
      answer_start: int, index into input_ids where the num_digits answer bytes begin
      answer_len  : int, == num_digits
      depth       : float, as given
      total_len   : int, as given (haystack length only)
    """
    passkey_sentence, digits = make_passkey_sentence(rng, num_digits)
    ps_len = len(passkey_sentence)

    if total_len < ps_len:
        # Degenerate case (very short haystack, e.g. tiny flash-tier configs): the
        # passkey sentence itself gets truncated. Documented, not hidden.
        body = passkey_sentence[:total_len]
    else:
        remaining = total_len - ps_len
        before_len = int(round(depth * remaining))
        before_len = max(0, min(before_len, remaining))
        after_len = remaining - before_len
        body = make_filler(before_len, rng) + passkey_sentence + make_filler(after_len, rng)

    assert len(body) == total_len, f"body length {len(body)} != total_len {total_len}"

    query_text = body + QUERY_PREFIX
    full_text = query_text + digits
    input_ids = np.frombuffer(full_text.encode("ascii"), dtype=np.uint8).astype(np.int64)

    return {
        "input_ids": input_ids,
        "answer_start": len(query_text),
        "answer_len": num_digits,
        "depth": depth,
        "total_len": total_len,
    }


def _collate(examples: list[dict], num_digits: int) -> dict:
    query_len = len(QUERY_PREFIX)
    tail_len = query_len + num_digits
    body_max = max(ex["total_len"] for ex in examples)
    l_max = body_max + tail_len
    b = len(examples)

    input_ids = np.zeros((b, l_max), dtype=np.int64)  # pad id = 0 (NUL)
    pad_mask = np.zeros((b, l_max), dtype=bool)
    targets = np.zeros((b, num_digits), dtype=np.int64)
    depths = np.zeros((b,), dtype=np.float32)
    lengths = np.zeros((b,), dtype=np.int64)

    for i, ex in enumerate(examples):
        body_len = ex["total_len"]
        pad_len = body_max - body_len
        seq = ex["input_ids"]
        input_ids[i, pad_len:pad_len + len(seq)] = seq
        pad_mask[i, pad_len:] = True
        targets[i] = seq[ex["answer_start"]:ex["answer_start"] + num_digits]
        depths[i] = ex["depth"]
        lengths[i] = body_len

    # Because every row's body is padded on the LEFT up to the common body_max, the
    # query+answer tail always sits in the last tail_len columns for every row.
    #
    # CRITICAL: answer_mask marks the positions whose LOGITS predict each answer digit,
    # NOT the positions where the digits themselves sit as input. Causal attention
    # includes self-attention (query position i can attend to key position i), so a
    # position whose own input token IS the target digit lets the model "predict" it by
    # trivially copying its own input embedding through the residual stream — a dead
    # giveaway if targets are read from the same positions they're fed in as input.
    # Confirmed as a real bug this way: feeding a model the correct example but with
    # garbage bytes written directly at the (then-unshifted) answer positions caused the
    # model to predict exactly that garbage, regardless of the actual passkey elsewhere
    # in the sequence — it had learned "echo my own input," not retrieval. The fix:
    # answer_mask marks the position BEFORE each digit (so its logits are computed from
    # context that does not yet include that digit, standard shifted next-token
    # supervision), matching the aux-LM-loss path in train.py (`logits[:, :-1]` vs.
    # `input_ids[:, 1:]`), which never had this bug because it was written with the
    # shift from the start.
    answer_mask = np.zeros((b, l_max), dtype=bool)
    answer_mask[:, l_max - num_digits - 1:l_max - 1] = True

    return {
        "input_ids": torch.from_numpy(input_ids),
        "targets": torch.from_numpy(targets),
        "answer_mask": torch.from_numpy(answer_mask),
        "pad_mask": torch.from_numpy(pad_mask),
        "depths": torch.from_numpy(depths),
        "lengths": torch.from_numpy(lengths),
    }


def make_batch(
    batch_size: int,
    len_range: tuple[int, int],
    rng: np.random.Generator,
    num_digits: int = 5,
    depth_range: tuple[float, float] = (0.0, 1.0),
) -> dict:
    """Mixed-length batch: each example's haystack length and needle depth sampled
    independently within the given ranges, then collated via left-padding the haystack
    (never the query/answer tail) so the answer_mask is identical across the batch."""
    examples = []
    for _ in range(batch_size):
        total_len = int(rng.integers(len_range[0], len_range[1] + 1))
        depth = float(rng.uniform(*depth_range))
        examples.append(build_example(total_len, depth, rng, num_digits))
    return _collate(examples, num_digits)


def make_eval_batch(length: int, depth: float, n_examples: int, rng: np.random.Generator, num_digits: int = 5) -> dict:
    """Fixed haystack length and depth for every example — used by the eval grid."""
    return make_batch(n_examples, (length, length), rng, num_digits, (depth, depth))


def make_inwindow_sanity_batch(window: int, n_examples: int, rng: np.random.Generator, num_digits: int = 5) -> dict:
    """The mandatory sanity-check batch: haystack length chosen so the FULL sequence
    (haystack + query + answer) is exactly `window` tokens, with depth varied uniformly
    across (0, 1) for coverage. See module docstring for why this differs from the
    eval grid's definition of "length"."""
    body_len = window - len(QUERY_PREFIX) - num_digits
    if body_len < 0:
        raise ValueError(
            f"window={window} is too small to fit QUERY_PREFIX ({len(QUERY_PREFIX)} chars) "
            f"+ {num_digits} answer digits; increase window."
        )
    return make_batch(n_examples, (body_len, body_len), rng, num_digits, (0.0, 1.0))


def decode_bytes(row: torch.Tensor | np.ndarray) -> str:
    """Decode a single row of byte ids back to text, for notebook display. NUL padding
    bytes are dropped; anything outside printable ASCII is shown as '.'."""
    if isinstance(row, torch.Tensor):
        row = row.cpu().numpy()
    chars = []
    for b in row.tolist():
        if b == 0:
            continue
        chars.append(chr(b) if 32 <= b < 127 else ".")
    return "".join(chars)

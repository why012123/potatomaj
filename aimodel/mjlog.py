"""Tenhou mjlog (VBA-clipped flavour) -> internal MJAI event stream.

Why this module exists
----------------------
A classmate supplied ``GL1.xlsm``: Sheet1 holds 40,000 Tenhou 4-player hands in
the official mjlog format, but the workbook's VBA stripped the outer tags. The
data restriction is strict -- local use only, never committed, never
redistributed -- so the parsed output goes to a gitignored directory.

Why convert to MJAI
-------------------
The replay/compare pipeline (``aimodel/logcheck.py``) and the training-data
extractor (``aimodel/logdata.py``) consume MJAI events. Converting once here
means neither of them needs to know anything about mjlog.

The clipped row format
----------------------
    " seed=0,0,0,2,2,13 ten=250,250,250,250 oya=0
      hai0=99,126,... hai1=... hai2=... hai3=...
      /><T46/><D114/>...<AGARI ... /><"

Differences from a real mjlog file, all handled below:

* the leading ``<INIT ...>`` tag is gone and the attributes are bare;
* ``haiN=`` has no surrounding tag;
* a row holds ONE hand, not a whole game, and it may end with a lone ``<``
  which actually belongs to the next row's ``<INIT``; some rows instead end
  with ``</mjloggm>`` (the last hand of a game). A draw row ends with the
  ``<RYUUKYOKU ... />`` tag followed by that same lone ``<``.

Note on the draw tag's spelling: it is ``RYUUKYOKU`` (two U's), the official
Tenhou mjlog spelling, not ``RYUKYOKU``. Both are accepted below.

Tile encoding
-------------
Tenhou uses 0..135 with four copies of each of the 34 tile types. The project's
own index is ``tenhou // 4`` (0-8 = 1m-9m, 9-17 = 1p-9p, 18-26 = 1s-9s,
27-33 = E S W N Haku Hatsu Chun). Red fives are Tenhou ids 16, 52, 88.

Call encoding
-------------
``m`` is a bit field (official mjlog layout -- see :func:`_decode_call`):

* bits 0-1: which seat the tile came from, counted FORWARD from the caller
  (1 = 下家 ``(seat + 1) % 4``, 2 = 対面, 3 = 上家, 0 = nobody, i.e. ankan);
* bit 2: 1 => chi;
* bit 3: 1 => pon; bit 4: 1 => kakan (added kan);
* bits 2-5 all 0: ``m & 3 == 0`` => ankan, otherwise daiminkan;
* chi: bits 3-4 / 5-6 / 7-8 are the copy subscripts of the run's low / middle /
  high tile and bits 10-15 are ``(suit * 7 + run) * 3 + called_pos``;
* pon / kakan: bits 5-6 are the copy that is NOT in the meld and bits 9-15 are
  ``tile * 3 + called_pos`` (pon only);
* ankan / daiminkan: bits 8-15 are ``tile * 4 + copy``.

Nothing here needs guessing: every call in ``GL1.xlsm`` decodes to a meld that
is consistent with the caller's hand, with the immediately preceding discard,
and with Tenhou's own ``AGARI m=`` ground truth (40,000 rows, 0 mismatches).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

#: Tenhou id -> project tile index
def t2p(tenhou_id: int) -> int:
    return int(tenhou_id) // 4


#: Tenhou red-five ids
RED_FIVES = {16, 52, 88}

#: call type field in the ``m=`` encoding
CALL_CHI, CALL_PON, CALL_KAN, CALL_SHOUMINKAN, CALL_ANKAN = 0, 1, 2, 3, 4
CALL_NAMES = {CALL_CHI: "chi", CALL_PON: "pon", CALL_KAN: "kan",
              CALL_SHOUMINKAN: "shouminkan", CALL_ANKAN: "ankan"}

#: the four per-seat tag families
TSUMO_TAGS = "TUVW"     # draw
DAHAI_TAGS = "DEFG"     # discard
SEAT_OF_TAG = {c: i for i, c in enumerate(TSUMO_TAGS)}
DAHAI_SEAT = {c: i for i, c in enumerate(DAHAI_TAGS)}

HONORS = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}

#: event tags we expect to see (anything else is recorded as unknown)
#: NOTE: the draw tag is spelled RYUUKYOKU (two U's) -- that is the official
#: Tenhou mjlog spelling and it is what the workbook actually contains: a full
#: scan of Sheet1's sharedStrings.xml finds 6,643 ``<RYUUKYOKU`` and exactly 0
#: ``<RYUKYOKU``. Accepting both keeps a differently-spelled source working.
RYUUKYOKU_TAGS = ("RYUUKYOKU", "RYUKYOKU")
KNOWN_TAGS = set("TUVWDEFG") | {"N", "REACH", "DORA", "AGARI", "BYE", "UN"} | set(RYUUKYOKU_TAGS)


class ParseError(Exception):
    """A row could not be turned into an event stream."""


# ---------------------------------------------------------------------------
# MJAI writing
#
# The task says to reuse ``aimodel/mjai.py``'s writer. That module is not
# present on this machine yet, so ``emit`` prefers it when it appears and falls
# back to the standard MJAI JSONL form otherwise. Both paths produce one JSON
# object per line, which is what MJAI logs are.
# ---------------------------------------------------------------------------


def _external_emitter():
    """Return ``aimodel.mjai``'s row-writer if that module exists, else None."""
    try:
        from . import mjai  # type: ignore
    except Exception:
        return None
    for attr in ("dumps", "to_json", "emit", "dumps_event"):
        fn = getattr(mjai, attr, None)
        if callable(fn):
            return fn
    return None


_EMIT = _external_emitter()


def emit(events: list[dict]) -> str:
    """Render events as MJAI log text (one JSON object per line)."""
    if _EMIT is not None:
        try:
            return "\n".join(_EMIT(e) for e in events)
        except Exception:
            pass  # fall back to the local writer rather than losing the row
    return "\n".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in events)


def writer_name() -> str:
    return "aimodel.mjai" if _EMIT is not None else "builtin"


# ---------------------------------------------------------------------------
# row splitting
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<([A-Za-z_]+)([^<>]*)>")
_ATTR_RE = re.compile(r"([A-Za-z_]+)\s*=\s*([^/>\s]+)")
#: value-carrying tags such as <T46/> <D114/> <E68/> -- no attributes, just a number
_BARE_RE = re.compile(r"(\d+)")


def _attrs(blob: str) -> dict[str, str]:
    return {k: v for k, v in _ATTR_RE.findall(blob or "")}


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x != ""]


def parse_row(row: str) -> list[dict]:
    """One clipped Sheet1 row -> a list of MJAI events.

    Raises :class:`ParseError` if the row does not look like a hand.
    """
    if row is None:
        raise ParseError("empty row")
    s = str(row).strip()
    if not s:
        raise ParseError("blank row")

    # The clipped row starts with " seed=" (no tag) and the hand data is bare.
    m = re.search(r"seed=([\d,]+)", s)
    if not m:
        raise ParseError("no seed=")
    seed = _ints(m.group(1))
    if len(seed) < 6:
        raise ParseError(f"seed has {len(seed)} fields, need 6")

    ten = _ints(re.search(r"ten=([\d,\-]+)", s).group(1)) if "ten=" in s else [250] * 4
    oya_m = re.search(r"oya=(\d+)", s)
    if not oya_m:
        raise ParseError("no oya=")
    oya = int(oya_m.group(1))

    hands: list[list[int]] = []
    raw_hands: list[list[int]] = []
    for seat in range(4):
        hm = re.search(rf"hai{seat}=([\d,]+)", s)
        if not hm:
            raise ParseError(f"missing hai{seat}=")
        ids = _ints(hm.group(1))
        if len(ids) != 13:
            raise ParseError(f"hai{seat} has {len(ids)} tiles, need 13")
        raw_hands.append(list(ids))
        hands.append([t2p(x) for x in ids])

    # Everything from the first event tag onwards is the event stream.
    first = re.search(r"/>|<[A-Z]", s)
    body = s[first.start():] if first else ""
    if not body:
        raise ParseError("no event stream")

    # ``seed`` layout (official spec):
    #   seed[0] = round+wind packed (0-3 = East 1-4, 4-7 = South, 8-11 = West)
    #   seed[1] = honba (NOT kyoku!), seed[2] = riichi sticks,
    #   seed[3], seed[4] = the two dice, seed[5] = the dora indicator
    bakaze_i, honba = seed[0] // 4, seed[1]
    kyoku = seed[0] % 4 + 1
    events: list[dict] = []
    events.append({
        "type": "start_kyoku",
        "bakaze": "ESWN"[bakaze_i % 4],
        "kyoku": kyoku,
        "honba": honba,
        "riichi_sticks": seed[2],
        "dice": [seed[3] + 1, seed[4] + 1],
        "scores": list(ten),
        "oya": oya,
        "dora_markers": [_name_of_raw(seed[5])] if len(seed) > 5 else [],
        "tehais": [[_name_of_raw(x) for x in raw] for raw in raw_hands],
    })
    # The INITIAL dora indicator must also be emitted as a ``dora`` event.
    # Reason: ``mjai.export`` writes start_kyoku and then one ``dora`` event per
    # indicator coming out of ``start_hand`` (engine.py: ``dora=[self._indicator(k)
    # ...]``) -- and logcheck.kyoku_to_wall reconstructs the dead wall ONLY from
    # ``{"type": "dora"}`` events (``kyoku_to_wall`` line: ``indicators = [... if
    # e.get("type") == "dora"]``). Emitting the seed[5] indicator only inside
    # start_kyoku therefore left ``dead[0]`` as filler in every hand without a
    # kan: measured on batch kyoku #1 the engine scored dora=0 for a hand whose
    # real indicator (29 -> 8m) made the 9m in hand a dora. Kan indicators
    # (``<DORA>`` tags) then land in dead[2], dead[4]... in the right order.
    if len(seed) > 5:
        events.append({"type": "dora", "dora_marker": _name_of_raw(seed[5])})

    # replay state
    hand_of: list[list[int]] = [list(h) for h in hands]
    melds_of: list[list[tuple]] = [[] for _ in range(4)]
    called_from: dict[int, int] = {}
    pond: list[list[int]] = [[] for _ in range(4)]
    last_tsumo: list[int | None] = [None] * 4
    last_tsumo_raw: list[int | None] = [None] * 4
    last_dahai: tuple[int, int] | None = None
    reach_declared = [False] * 4
    dora_count = 0

    #: seats whose riichi declaration was actually accepted (they have a
    #: ``<REACH ... step="2">`` tag). A step="1" without a step="2" means the
    #: player was ronned on the declaring discard: Tenhou charges no stick, so the
    #: ``reach`` event must not be emitted (see the REACH branch below).
    accepted_reach = {
        int(b2.get("who", -1))
        for t2, blob2 in _TAG_RE.findall(body)
        if t2 == "REACH"
        for b2 in [_attrs(blob2)]
        if b2.get("step", "1") == "2"
    }

    for tag, blob in _TAG_RE.findall(body):
        a = _attrs(blob)

        if tag in SEAT_OF_TAG:  # draw
            seat = SEAT_OF_TAG[tag]
            raw = _bare_raw(blob, tag, s)
            tile = t2p(raw)
            last_tsumo[seat] = tile
            last_tsumo_raw[seat] = raw
            # the drawn tile joins the hand; the next discard removes it again
            hand_of[seat].append(tile)
            events.append({"type": "tsumo", "actor": seat, "pai": _name_of_raw(raw)})
            continue

        if tag in DAHAI_SEAT:  # discard
            seat = DAHAI_SEAT[tag]
            raw = _bare_raw(blob, tag, s)
            tile = t2p(raw)
            # tsumogiri must compare RAW ids: two different 5p exist (52 = red),
            # so index equality would call a tedashi of the other 5p a tsumogiri.
            tsumogiri = last_tsumo_raw[seat] == raw
            # the tile leaves the hand; red-five identity comes from the raw id
            _remove_tile(hand_of[seat], tile)
            pond[seat].append(tile)
            last_dahai = (seat, tile)
            ev = {"type": "dahai", "actor": seat, "pai": _name_of_raw(raw),
                  "tsumogiri": tsumogiri}
            if seat in called_from:
                ev["_after_call"] = called_from.pop(seat)
            events.append(ev)
            continue

        if tag == "N":
            seat = int(a.get("who", -1))
            mv = int(a.get("m", -1))
            if seat < 0 or mv < 0:
                raise ParseError(f"bad N tag: {blob!r}")
            ev, consumed = _decode_call(mv, seat, last_dahai, hand_of)
            events.append(ev)
            # ``consumed`` = every tile this call takes OUT OF THE CALLER'S HAND
            # (for ankan that is all four; the taken discard is not included).
            for t in consumed:
                _remove_tile(hand_of[seat], t)
            if ev["type"] == "chi":
                melds_of[seat].append(("chi", tuple([_pai_index(ev["pai"])] + consumed)))
            elif ev["type"] == "pon":
                melds_of[seat].append(("pon", tuple([_pai_index(ev["pai"])] + consumed)))
            else:
                melds_of[seat].append((ev["type"],
                                       tuple([_pai_index(ev["pai"])] + consumed)))
            if ev.get("_from") is not None:
                called_from[seat] = ev.pop("_from")
                if last_dahai is not None and last_dahai[0] != seat:
                    tile = last_dahai[1]
                    if pond[last_dahai[0]] and pond[last_dahai[0]][-1] == tile:
                        pond[last_dahai[0]].pop()  # taken into the meld
            continue

        if tag == "REACH":
            seat = int(a.get("who", -1))
            step = a.get("step", "1")
            if step == "1":
                reach_declared[seat] = True
                # A declaration whose discard is ronned never gets a step="2":
                # the riichi is NOT established and Tenhou never charges the 1000.
                # Proof from sheet row 2: seats 1 and 0 both emit <REACH step=1>,
                # only seat 1 has step=2, and the AGARI before-scores are
                # 237,233,233,287 against a start of 237,243,233,287 -- exactly one
                # 1000 deduction. Emitting ``reach`` here made the replaying engine
                # charge a stick it should not have (logcheck batch kyoku #1:
                # engine paid the winner +14000 where the log says +13000).
                if seat not in accepted_reach:
                    continue
                events.append({"type": "reach", "actor": seat})
            else:
                ev = {"type": "reach_accepted", "actor": seat}
                if "ten" in a:
                    ev["scores"] = _ints(a["ten"])
                events.append(ev)
            continue

        if tag == "DORA":
            dora_count += 1
            # mjai.export writes {"type": "dora", "dora_marker": <tile>} (singular
            # string) and logcheck.kyoku_to_wall reads e["dora_marker"], so match
            # that exactly. a["hai"] is a raw Tenhou id string, so convert first.
            events.append({"type": "dora",
                           "dora_marker": _name_of_raw(int(a["hai"]))})
            continue

        if tag == "AGARI":
            events.append(_decode_agari(a, hand_of, melds_of, oya))
            # A hand must be CLOSED with end_kyoku: mjai.iter_kyoku only yields a
            # kyoku when it sees one, so without this logcheck silently compares
            # almost nothing (measured: 497 hands parsed -> only 8 compared).
            events.append({"type": "end_kyoku", "reason": "hora"})
            continue

        if tag in RYUUKYOKU_TAGS:
            events.append(_decode_ryukyoku(a, hand_of, melds_of, ten))
            events.append({"type": "end_kyoku", "reason": "ryukyoku"})
            continue

        if tag in ("BYE", "UN"):
            events.append({"type": "end_kyoku", "reason": tag.lower()})
            continue

        # unknown tag: keep the stream honest instead of dropping it silently
        events.append({"type": "_unknown", "tag": tag, "raw": blob.strip()})

    return events


# ---------------------------------------------------------------------------
# tile naming / bookkeeping
# ---------------------------------------------------------------------------

_SUITS = "mps"
_NUM = "123456789"
#: honors, project indices 27..33 = E S W N + Haku Hatsu Chun
_HONOR = "ESWNPFC"


def _tile_name(t: int) -> str:
    """Project tile index -> MJAI tile string ("5m", "5mr", "E", ...)."""
    if t < 0 or t > 33:
        return "?"
    if t >= 27:
        return _HONOR[t - 27]
    suit = t // 9
    num = t % 9
    return _NUM[num] + _SUITS[suit]


def _red_name(t: int) -> str:
    """Project index of a *red five* -> MJAI's red spelling ``"0m"/"0p"/"0s"``."""
    return "0" + _SUITS[t // 9]


def _name_of_raw(tenhou_id: int) -> str:
    """Raw Tenhou tile id -> MJAI name, **keeping red-five identity**.

    Tenhou ids 16 / 52 / 88 are the red fives. ``t2p`` maps them onto the same
    project index as an ordinary five (4 / 13 / 22), so calling ``_tile_name``
    directly loses that bit and every hand containing a red five silently loses
    its aka-dora han (measured: sheet row 1 -- Tenhou scores ``ten=20,2700,0``
    with ``yaku=0,1,7,1,54,1``, i.e. 3 han including aka; the engine fed ``"5p"``
    scored 1 han). MJAI spells a red five ``0m``/``0p``/``0s`` and
    ``mjai.tile_from_mjai`` maps that to ``(index, red=True)``, which is what
    ``mjai.export``/``hand_to_mjai`` write, so this is the project's own
    convention -- not a new one.
    """
    t = t2p(tenhou_id)
    if tenhou_id in RED_FIVES and 0 <= t <= 33:
        return _red_name(t)
    return _tile_name(t)


#: the T/U/V/W and D/E/F/G tags carry their value as the tag body, not an attr
_TAGVAL_RE = {c: re.compile(rf"<{c}(\d+)/>") for c in "TUVWDEFG"}

#: MJAI tile name -> project index (inverse of _tile_name)
_NAME_TO_INDEX = {_tile_name(i): i for i in range(34)}
#: red fives spell the same tile type, so they index identically
for _i in (4, 13, 22):
    _NAME_TO_INDEX[_red_name(_i)] = _i


def _pai_index(pai) -> int:
    """'5p' / '0p' -> 13. Accepts an int unchanged (for internal bookkeeping)."""
    if isinstance(pai, int):
        return pai
    return _NAME_TO_INDEX.get(str(pai), -1)


def _bare_raw(blob: str, tag: str, row: str) -> int:
    """Value of a value-carrying tag as the RAW Tenhou id: ``<T46/>`` -> 46.

    ``blob`` is what _TAG_RE captured between the tag name and the closing angle
    bracket, e.g. ``"46/"`` or ``"114/"``. Take the FIRST number in it: falling
    back to a whole-row regex search is wrong, because ``<E125/>`` would then
    match some other tag's digits (125//4 = 31 Haku, not 18).

    The raw id is returned (not the project index) because it is the only place
    the red-five bit survives: 52 -> ``0p``, 53 -> ``5p``.
    """
    m = _BARE_RE.search(blob or "")
    if m:
        return int(m.group(1))
    m = _TAGVAL_RE[tag].search(row) if tag in _TAGVAL_RE else None
    if not m:
        raise ParseError(f"tag <{tag}> has no value")
    return int(m.group(1))


def _bare_tile(blob: str, tag: str, row: str) -> int:
    """As :func:`_bare_raw` but returning the project tile index."""
    return t2p(_bare_raw(blob, tag, row))


def _tag_value(tag: str, row: str) -> int:
    m = _TAGVAL_RE[tag].search(row)
    if not m:
        raise ParseError(f"tag <{tag}> has no value")
    return t2p(m.group(1))


def _remove_tile(hand: list[int], tile: int) -> None:
    try:
        hand.remove(tile)
    except ValueError:
        pass


def _decode_call(mv: int, seat: int, last_dahai, hand_of):
    """Decode ``m=`` into an MJAI call event plus the tiles the caller gives up.

    Returns ``(event, consumed)``; ``consumed`` holds the project tile indices
    the caller gives up (never the called tile itself -- ankan gives up four).
    ``_from`` is set when the tile came from another seat, so the caller can pop
    it out of that player's pond.

    Every field is taken from the official bit layout (module docstring), so the
    chi's run *and* which of its three tiles was called, the pon's tile, and the
    ankan/daiminkan tile all come out of ``m``.  The old code instead guessed the
    meld from the caller's hand, and guessed wrong whenever the hand could back
    two different runs for the same called tile: it then removed the wrong two
    tiles, so a later discard of one of them silently removed nothing and every
    following call was judged against a hand that had drifted.
    """
    low = mv & 3
    hand = hand_of[seat]
    # bits 0-1 name the seat the tile came from, counted FORWARD from the caller
    # (1 = the next player, 3 = the previous one -- the only seat a chi may take
    # from; 0 = nobody, i.e. ankan).  In every chi / pon / daiminkan of GL1.xlsm
    # this agrees with the tile's own immediately preceding discard.
    source = (seat + low) % 4
    taken = last_dahai[1] if last_dahai else None

    # -- chi (bit 2) ---------------------------------------------------------
    if mv & 0x04:
        pat = (mv >> 10) & 0x3F              # (suit * 7 + run) * 3 + called_pos
        pos, suit_run = pat % 3, pat // 3
        suit, run = suit_run // 7, suit_run % 7
        seq = [suit * 9 + run + k for k in range(3)]
        tile = seq[pos]                      # the tile that was called
        consumed = [t for k, t in enumerate(seq) if k != pos]
        missing = [_tile_name(t) for t in consumed if t not in hand]
        if missing:
            raise ParseError(f"chi of {_tile_name(tile)} (m={mv}) but seat {seat} "
                             f"holds no {'/'.join(missing)}")
        if taken is None:
            raise ParseError(f"call m={mv} at seat {seat} but no preceding discard")
        return ({"type": "chi", "actor": seat, "target": source,
                 "pai": _tile_name(tile),
                 "consumed": [_tile_name(t) for t in consumed],
                 "_from": source},
                consumed)

    # -- pon (bit 3) / kakan (bit 4) -----------------------------------------
    if mv & 0x18:
        tile = ((mv >> 9) & 0x7F) // 3       # == (m >> 4) // 96
        have = sum(1 for t in hand if t == tile)
        if mv & 0x08:
            if have < 2:
                raise ParseError(f"pon of {_tile_name(tile)} (m={mv}) but seat "
                                 f"{seat} holds only {have}")
            if taken is None:
                raise ParseError(f"call m={mv} at seat {seat} but no preceding discard")
            return ({"type": "pon", "actor": seat, "target": source,
                     "pai": _tile_name(tile), "consumed": [_tile_name(tile)] * 2,
                     "_from": source},
                    [tile] * 2)
        # kakan: the fourth copy joins the pon this seat already has
        if have < 1:
            raise ParseError(f"kakan of {_tile_name(tile)} (m={mv}) but seat {seat} "
                             f"does not hold the fourth copy")
        return ({"type": "kakan", "actor": seat, "pai": _tile_name(tile),
                 "consumed": [], "_from": None}, [tile])

    # -- ankan / daiminkan (bits 2-5 clear) ----------------------------------
    tile = ((mv >> 8) & 0xFF) // 4
    have = sum(1 for t in hand if t == tile)
    if low == 0:                             # ankan: no tile was called
        if have < 4:
            raise ParseError(f"ankan of {_tile_name(tile)} (m={mv}) but seat {seat} "
                             f"holds only {have}")
        return ({"type": "ankan", "actor": seat, "pai": _tile_name(tile),
                 "consumed": [_tile_name(tile)] * 3, "_from": None}, [tile] * 4)
    if have < 3:
        raise ParseError(f"daiminkan of {_tile_name(tile)} (m={mv}) but seat {seat} "
                         f"holds only {have}")
    if taken is None:
        raise ParseError(f"call m={mv} at seat {seat} but no preceding discard")
    return ({"type": "minkan", "actor": seat, "target": source,
             "pai": _tile_name(tile), "consumed": [_tile_name(tile)] * 3,
             "_from": source}, [tile] * 3)


def _chi_shape(hand: list[int], tile: int, payload: int):
    """Return the three project indices of a chi containing ``tile``."""
    if tile >= 27:
        return None
    num = tile % 9
    suit = tile // 9
    have = set(hand)
    # the three possible runs containing `tile`
    cands = []
    for start in (num - 2, num - 1, num):
        if start < 0 or start + 2 > 8:
            continue
        seq = [suit * 9 + start + k for k in range(3)]
        if all(t in have for t in seq if t != tile):
            cands.append(seq)
    if not cands:
        return None
    return cands[payload % len(cands)]


# ---------------------------------------------------------------------------
# AGARI / RYUKYOKU
# ---------------------------------------------------------------------------


def _decode_agari(a: dict, hand_of, melds_of, oya: int) -> dict:
    who = int(a.get("who", -1))
    from_who = int(a.get("fromWho", who))
    machi_raw = int(a["machi"]) if "machi" in a else None
    machi = t2p(machi_raw) if machi_raw is not None else None
    raw_hai = _ints(a.get("hai", ""))
    hai = [t2p(x) for x in raw_hai]
    ten = _ints(a.get("ten", ""))          # [fu, points, (limit tier)]
    yaku = _ints(a.get("yaku", ""))        # pairs (id, han)
    sc = _ints(a.get("sc", ""))
    ba = _ints(a.get("ba", ""))
    deltas = _sc_deltas(sc)
    # ``ten[2]`` is NOT a yakuman flag -- it is the *limit tier* of the hand.
    # Measured over all 5,012 AGARI rows of Sheet1 (points seen per tier):
    #   0 -> 1000..7700      1 -> 8000/12000 (mangan)
    #   2 -> 12000/18000     3 -> 16000/24000 (baiman)
    #   4 -> 24000           5 -> 32000/48000 (yakuman; those rows log yaku="")
    # So a mangan (tier 1) would have been reported as "yakuman" by the old key.
    tier = ten[2] if len(ten) > 2 else 0
    return {
        "type": "hora",
        "actor": who,
        "target": from_who,
        "tsumo": who == from_who,
        "pai": _name_of_raw(machi_raw) if machi_raw is not None else None,
        "oya": oya,
        "fu": ten[0] if ten else None,
        "points": ten[1] if len(ten) > 1 else None,
        #: 0 = plain, 1 = mangan, 2 = haneman, 3 = baiman, 4 = sanbaiman, 5 = yakuman
        "limit_tier": tier,
        "yakuman_flag": 1 if tier == 5 else 0,
        "yaku_ids": yaku,
        "han_ids": yaku[1::2],
        #: total han as logged (includes dora 52 / ura 53 / aka 54 entries).
        #: logcheck.check_kyoku only compares han when the key is present.
        "han": sum(yaku[1::2]),
        "hand_concealed": [_name_of_raw(x) for x in raw_hai],
        "n_melds": len(melds_of[who]) if who >= 0 else 0,
        "deltas": deltas,
        "raw_sc": sc,
        "ba": ba,
        "dora_markers": [_name_of_raw(x) for x in _ints(a.get("doraHai", ""))],
        "ura_markers": [_name_of_raw(x) for x in _ints(a.get("doraHaiUra", ""))],
    }


_RYUKYOKU_TYPE = {
    "": "exhaustive_draw",
    "nm": "exhaustive_draw",
    "yao9": "kyuushu_kyuuhai",
    "kaze4": "four_winds",
    "reach4": "four_reach",
    "ron3": "three_ron",
    "kan4": "four_kan",
}


def _sc_deltas(sc: list[int]) -> list[int]:
    """Tenhou ``sc`` -> per-seat point deltas.

    ``sc`` is ``[score0, var0, score1, var1, ...]``: the *score* followed by the
    *variation*, and the variation is in units of 100 points. Verified against
    5,012 real AGARI rows in GL1.xlsm:

        sum(var*100) == 1000 * sticks            -- 5012/5012 = 100.0%
        winner var*100 == ten[1] + 300*honba + 1000*sticks
                       -- tsumo 2758/2758 = 100.0%, ron 3914/3926 = 99.7%

    The deltas therefore INCLUDE the winner's collected riichi sticks and honba
    bonus, matching how ``engine.py`` settles (``Game._take_sticks`` adds
    ``sticks * riichi_stick`` into the winner's delta). NOTE: ``engine.py`` has
    no honba concept at all, so hands with ``honba > 0`` will still differ --
    that is an engine rules gap, reported rather than papered over.

    The previous code did ``sc[i+1] - sc[i]``, subtracting a *score* from a
    *variation* -- two different quantities. That fabricated the payments fed to
    the engine and produced bogus "点数收支不符" reports.
    """
    deltas = []
    for i in range(0, min(len(sc), 8), 2):
        if i + 1 < len(sc):
            deltas.append(sc[i + 1] * 100)
    return deltas


def _decode_ryukyoku(a: dict, hand_of, melds_of, ten: list[int]) -> dict:
    t = a.get("type", "")
    sc = _ints(a.get("sc", ""))
    deltas = _sc_deltas(sc)
    # ``sc`` is 8 numbers = (score, variation) per seat, the variation in units
    # of 100 points -- NOT (before, after). A noten transfer reads
    # sc=248,-15,228,15,335,-15,179,15 -> deltas [-1500,+1500,-1500,+1500],
    # which is the correct ±1500 four-player noten penalty; reading it as
    # (before, after) would give a nonsensical ±15 points.
    #
    # ``tenpai`` is deliberately left as-is: it is empty for these clipped rows
    # because the workbook puts the draw's ``haiN=`` tenpai hands *after* the
    # tag's ``/>``, not in the attribute blob this function receives. Decoding
    # it needs the row text and a signature change, so it is left alone rather
    # than guessed at.
    return {
        "type": "ryukyoku",
        "reason": _RYUKYOKU_TYPE.get(t, t or "exhaustive_draw"),
        "raw_type": t,
        "deltas": deltas,
        "raw_sc": sc,
        "tenpai": [_ints(a[k]) if k in a else [] for k in ("hai0", "hai1", "hai2", "hai3")],
    }


# ---------------------------------------------------------------------------
# sheet reading / dumping
# ---------------------------------------------------------------------------


def parse_sheet(path: str | Path, sheet: str = "Sheet1",
                limit: int | None = None) -> Iterator[tuple[int, str]]:
    """Yield ``(row_index, row_text)`` from the xlsm sheet, 1-based."""
    try:
        import openpyxl
    except ImportError as e:  # pragma: no cover
        raise SystemExit("openpyxl is required to read the workbook: " + str(e))

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    try:
        ws = wb[sheet]
        for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if limit is not None and i > limit:
                break
            v = row[0] if row else None
            if v is None:
                continue
            yield i, str(v)
    finally:
        wb.close()


def dump(path: str | Path, out_dir: str | Path, sheet: str = "Sheet1",
         limit: int | None = None, batch: int = 500,
         batching: str = "count", verbose: bool = True) -> dict:
    """Parse the sheet and write MJAI files into ``out_dir`` in batches.

    ``batching="count"`` (default) writes fixed-size chunks of ``batch`` hands
    per file -- 40,000 rows become ~80 files instead of 40,000. One file per row
    is useless here and floods the filesystem.

    ``batching="game"`` instead groups consecutive rows by ``seed[0]``, the
    encoded round index. Note that ``seed[0]`` is per-HAND (it counts 東1, 東2,
    ... within a game), not per-match, so this does NOT reliably reunite the
    hands of one game -- ``count`` is what you want unless you know the source
    is ordered by match.

    Returns a report dict with success/failure counts and failure samples.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = n_events = 0
    failures: list[dict] = []
    files: list[str] = []
    buf: list[str] = []
    buf_key = None
    buf_n = 0
    idx = 0

    def flush(key):
        nonlocal buf, buf_n
        if not buf:
            return
        # ".json" not ".jsonl": logcheck.py discovers inputs with glob("*.json"),
        # and the on-disk layout is the same either way (one JSON object per line).
        name = f"{key}.json"
        (out / name).write_text("\n".join(buf) + "\n", encoding="utf-8")
        files.append(name)
        buf = []
        buf_n = 0

    for row_i, row in parse_sheet(path, sheet=sheet, limit=limit):
        idx += 1
        try:
            events = parse_row(row)
        except ParseError as e:
            n_bad += 1
            if len(failures) < 200:
                failures.append({"row": row_i, "error": str(e),
                                 "head": row[:160], "len": len(row)})
            continue
        except Exception as e:  # unexpected: record, keep going
            n_bad += 1
            if len(failures) < 200:
                failures.append({"row": row_i, "error": f"{type(e).__name__}: {e}",
                                 "head": row[:160], "len": len(row)})
            continue

        n_ok += 1
        n_events += len(events)
        text = emit(events)
        game_key = _game_key(row)
        if batching == "count":
            buf.extend(text.splitlines())
            buf_n += 1
            if buf_n >= batch:
                flush(f"batch_{idx:06d}")
        else:
            if buf_key is None:
                buf_key = f"game_{idx:06d}_{game_key}"
            elif game_key != buf_key.split("_")[-1]:
                flush(buf_key)
                buf_key = f"game_{idx:06d}_{game_key}"
            buf.extend(text.splitlines())
            buf_n += 1
    flush(buf_key or f"game_{idx:06d}")

    report = {
        "rows_parsed": n_ok + n_bad,
        "ok": n_ok,
        "failed": n_bad,
        "fail_rate": (n_bad / (n_ok + n_bad)) if (n_ok + n_bad) else 0.0,
        "events": n_events,
        "files": len(files),
        "out_dir": str(out),
        "writer": writer_name(),
        "failure_samples": failures[:5],
    }
    if verbose:
        print_report(report)
    return report


_GAME_KEY_RE = re.compile(r"seed=(\d+),(\d+),")


def _game_key(row: str) -> str:
    """Identify the game a hand belongs to (used to group hands into files)."""
    m = _GAME_KEY_RE.search(str(row))
    return f"{m.group(1)}_{m.group(2)}" if m else "unknown"


def print_report(r: dict) -> None:
    print("=" * 66)
    print("  mjlog parse report")
    print("=" * 66)
    print(f"  rows parsed     {r['rows_parsed']:,}")
    print(f"  ok              {r['ok']:,}")
    print(f"  failed          {r['failed']:,}   ({r['fail_rate']:.3%})")
    print(f"  events emitted  {r['events']:,}")
    print(f"  mjai files      {r['files']:,}  ->  {r['out_dir']}")
    print(f"  writer          {r['writer']}")
    print("=" * 66)
    for s in r.get("failure_samples", []):
        print(f"  row {s['row']}: {s['error']}")
        print(f"      len={s['len']}  {s['head']!r}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Tenhou mjlog -> MJAI")
    ap.add_argument("path", help="GL1.xlsm")
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--out", default="aimodel/mjai_dump")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--batching", choices=["game", "count"], default="count")
    ap.add_argument("--report", default=None, help="write the report JSON here")
    args = ap.parse_args(argv)

    rep = dump(args.path, args.out, sheet=args.sheet, limit=args.limit,
               batch=args.batch, batching=args.batching)
    if args.report:
        Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

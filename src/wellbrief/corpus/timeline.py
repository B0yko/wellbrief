"""A well's operations on a clock, cut into daily reports.

The simulation plans a well as one queue of operations: drilling to a depth,
moving the string (tripping, running casing, reaming), fixed-length jobs
(circulating, cementing, waiting on cement, testing the BOP) and NPT events.
A daily report is the next 24 hours of that queue, 06:00 to 06:00. An
operation still running at 06:00 is split: the report gets its share, and
the rest stays at the head of the queue for the next report. Nothing is
squeezed to fit a day, so trips, casing jobs and long NPT events take the
time they take.

Hours are integer tenths throughout (one tenth is 6 minutes on the
operations log), so every report adds up to exactly 24.0 h.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

from .fields import FieldSpec, SectionPlan
from .records import Activity, NptEntry, NptEvent, fmt_m

DAY_TENTHS = 240


@dataclass(eq=False)
class Op:
    """One operation of the well plan.

    `drill` deepens the hole from `a` to `b`; `move` moves the bit (or a
    casing string) from `a` to `b` at `speed` m/h, 0 being surface; `fixed`
    takes `tenths`; `fill` takes the rest of the report; `npt` is `event`.
    """

    kind: str
    section: SectionPlan
    text: str = ""
    tenths: int = 0
    a: int = 0
    b: int = 0
    speed: float = 0.0
    event: NptEvent | None = None
    sets_mw: float | None = None     # mud weight in the hole from the start of this operation
    sets_mud: str | None = None      # mud system in the hole from the start of this operation
    tag: str = ""
    hint: str = ""                   # the "Next 24 h" remark while this operation is next
    started: bool = False


@dataclass
class Segment:
    """The part of an operation that falls inside one report."""

    op: Op
    start: int          # tenths after 06:00
    tenths: int
    hole_start: int
    hole_end: int
    a: int              # drill: hole depth from; move: string depth from
    b: int
    done: bool          # the operation ends inside this report


def drill_hours(spec: FieldSpec, start: float, end: float, factor: float) -> float:
    """Hours of drilling from `start` to `end` (the rate follows the formation)."""
    depth, hours = float(start), 0.0
    while depth < end - 1e-9:
        f = spec.formation_at(depth)
        rate = f.rop_m_per_day * factor / 24.0
        stop = end if f is spec.formations[-1] else min(end, float(f.base_m))
        hours += (stop - depth) / rate
        depth = stop
    return hours


def drill_reach(spec: FieldSpec, start: float, hours: float, factor: float) -> float:
    """Depth reached after drilling for `hours` from `start`."""
    depth, remaining = float(start), hours
    while remaining > 1e-9:
        f = spec.formation_at(depth)
        rate = f.rop_m_per_day * factor / 24.0
        needed = math.inf if f is spec.formations[-1] else (f.base_m - depth) / rate
        if needed >= remaining:
            return depth + remaining * rate
        depth = float(f.base_m)
        remaining -= needed
    return depth


def _ceil_tenths(hours: float) -> int:
    return max(1, math.ceil(hours * 10 - 1e-9))


def plan_segments(queue: list[Op], hole: int, spec: FieldSpec, factor: float) -> list[Segment]:
    """Lay the head of the queue out over the next 24 hours, without changing it."""
    segs: list[Segment] = []
    t = 0
    for op in queue:
        left = DAY_TENTHS - t
        if left <= 0:
            break
        if op.kind == "drill":
            need = _ceil_tenths(drill_hours(spec, op.a, op.b, factor))
            if need <= left:
                seg = Segment(op, t, need, hole, op.b, op.a, op.b, True)
            else:
                reach = max(op.a + 1, math.floor(drill_reach(spec, op.a, left / 10, factor)))
                seg = Segment(op, t, left, hole, min(reach, op.b), op.a, min(reach, op.b), reach >= op.b)
            hole = seg.hole_end
        elif op.kind == "move":
            dist = abs(op.b - op.a)
            need = _ceil_tenths(dist / op.speed)
            if need <= left:
                seg = Segment(op, t, need, hole, hole, op.a, op.b, True)
            else:
                step = dist * left // need
                mid = op.a + step if op.b > op.a else op.a - step
                seg = Segment(op, t, left, hole, hole, op.a, mid, False)
        elif op.kind in ("fixed", "npt"):
            used = min(op.tenths, left)
            seg = Segment(op, t, used, hole, hole, op.a, op.b, used == op.tenths)
        elif op.kind == "fill":
            seg = Segment(op, t, left, hole, hole, op.a, op.b, True)
        else:
            raise ValueError(f"unknown operation kind {op.kind!r}")
        segs.append(seg)
        t += seg.tenths
    return segs


def insert_at_depth(queue: list[Op], depth: int, new_ops: list[Op]) -> None:
    """Put `new_ops` where the bit reaches `depth` while drilling."""
    for i, op in enumerate(queue):
        if op.kind != "drill" or not op.a <= depth <= op.b:
            continue
        if depth == op.a:
            queue[i:i] = new_ops
        elif depth == op.b:
            queue[i + 1:i + 1] = new_ops
        else:
            queue[i:i] = [replace(op, b=depth), *new_ops]
            op.a = depth
        return
    raise ValueError(f"no drilling operation reaches {depth} m")


def insert_at_time(queue: list[Op], segs: list[Segment], t: int,
                   make: Callable[[SectionPlan, int, bool], list[Op]]) -> None:
    """Put the operations `make(section, depth, drilling)` returns at `t` tenths into the planned day.

    While drilling, the event happens at the depth the bit has reached by
    then. During an NPT event it waits until the event is over. Otherwise it
    happens before the operation that is running at `t`.
    """
    seg = next((s for s in segs if s.start <= t < s.start + s.tenths), segs[-1])
    op = seg.op
    if op.kind == "drill":
        depth = seg.hole_start + (seg.hole_end - seg.hole_start) * (t - seg.start) // seg.tenths
        insert_at_depth(queue, depth, make(op.section, depth, True))
        return
    i = next(i for i, o in enumerate(queue) if o is op)
    if op.kind == "npt":
        queue[i + 1:i + 1] = make(op.section, seg.hole_end, False)
    else:
        queue[i:i] = make(op.section, seg.hole_start, False)


def place(depth: int) -> str:
    return "surface" if depth == 0 else f"{fmt_m(depth)} m"


def run_segments(queue: list[Op], segs: list[Segment], report_id: str,
                 on_start: Callable[[Op], None]) -> list[Activity]:
    """Consume the planned segments from the queue and write the operations log."""
    acts: list[Activity] = []
    for seg in segs:
        op = seg.op
        if not op.started:
            op.started = True
            on_start(op)
        if op.kind == "drill":
            text = f"Drilled {op.section.size} hole from {fmt_m(seg.a)} m to {fmt_m(seg.b)} m."
            op.a = seg.b
        elif op.kind == "move":
            text = op.text.format(a=place(seg.a), b=place(seg.b))
            op.a = seg.b
        elif op.kind == "npt":
            assert op.event is not None
            ev = op.event
            first = not ev.parts
            which = ("whole" if seg.done else "first") if first else ("last" if seg.done else "middle")
            entry = NptEntry(ev, len(ev.parts) + 1, seg.tenths, ev.text(which, seg.tenths), report_id)
            ev.parts.append(entry)
            op.tenths -= seg.tenths
            acts.append(Activity(f"NPT {ev.code} {seg.tenths / 10:.1f} h: {entry.description}", seg.tenths,
                                 entry, op.kind, op.tag))
            if seg.done:
                _pop(queue, op)
            continue
        else:
            text = op.text
            op.tenths -= seg.tenths
        acts.append(Activity(text, seg.tenths, None, op.kind, op.tag))
        if seg.done:
            _pop(queue, op)
    return acts


def _pop(queue: list[Op], op: Op) -> None:
    if queue[0] is not op:
        raise RuntimeError("operations must finish in queue order")
    queue.pop(0)

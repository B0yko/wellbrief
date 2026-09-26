"""Simulation of one well: the plan, the day's events and the daily reports.

A well is planned as one queue of operations (see `timeline.py`):

- every hole section starts with its bottom-hole assembly, a trip in, the
  drill-out of the previous casing shoe with a change to the section's mud,
  and a formation integrity test 3 m below the shoe;
- drilling to the section base is cut where something happens at a known
  depth: the weight-up before Keldra Salt, saturated brine sweeps, the LCM
  pill above Vessra Carbonate, the total losses entering it, the first
  failure of a PJ-3 MWD in hole hotter than its limit;
- each section ends with its casing sequence (circulate, trip out, run and
  cement casing, wait on cement, test the BOP), and the last one with
  logging, the liner and the rig-down.

The one-off planted events are part of that plan, so they happen at most
once per well: the pack-off on the trip out for the intermediate casing
(and the fishing job that may follow it), and the total losses on entering
the carbonate. The first MWD failure is planned the same way; later ones are
drawn day by day.

Each daily report then draws the day's repeatable events from what the plan
says the day will do: salt creep while drilling the salt, partial losses
below the carbonate top, repeat MWD failures in the hot 12 1/4" hole, mud
pump repairs on one rig and background NPT. An event tied to a depth happens when
the bit gets there; the others interrupt whatever is running at a random
time. Only then is the day laid out, so every depth and duration on a report
comes from the same timeline.
"""

from __future__ import annotations

import math
import random

from .events import NOISE_ANY, NOISE_DRILLING, NOISE_PROBABILITY, VARIANTS
from .fields import (
    CARBONATE,
    G2_CROSSING_M,
    G2_PARTIAL_BASE_M,
    G3_BHT_LIMIT_C,
    G3_SECTION,
    G3_TOOL,
    G4_RIG,
    MWD_VENDOR,
    SALT,
    Formation,
    SectionPlan,
)
from .records import Activity, DayReport, NptEvent, WellRecord, fmt_m
from .timeline import DAY_TENTHS, Op, Segment, insert_at_depth, insert_at_time, plan_segments, run_segments

MAX_REPORTS_PER_WELL = 200

G1_PACKOFF_P = 0.80          # a low mud weight well packs off on the trip out for casing
G1_FISHING_P = 0.28          # the string stays stuck and is fished
G1_CREEP_P = 0.34            # per report drilling the salt at low mud weight
G2_PARTIAL_P = 0.55          # per report drilling the first 120 m of carbonate after total losses
G3_FIRST_FAILURE_P = 0.80    # a PJ-3 fails somewhere in the hole hotter than the limit ...
G3_REPEAT_P = 0.10           # ... and a replacement module again on a later report drilling that hole
G4_P = 0.16                  # per report on the rig with the worn pump
CEMENT_ISSUE_P = 0.20        # per cement job
BOP_FAILURE_P = 0.15         # per BOP test after a casing string
STANDBY_P = 0.12             # a third-party crew is late for a casing run or for logging

BITS = ("M323", "M433", "S123", "M223")
DRILLING_PARAMETERS = {       # WOB t, RPM, flow gpm
    '26"': ((5, 15), (60, 100), (900, 1150)),
    '17 1/2"': ((10, 22), (100, 150), (750, 1000)),
    '12 1/4"': ((8, 20), (110, 160), (550, 800)),
    '8 1/2"': ((6, 14), (120, 170), (480, 600)),
}
PRETREAT_FLOW_GPM = 430
PRETREAT_INTERVAL_M = 60     # the reduced flow rate holds for the first 60 m of carbonate


def tenths(hours: float) -> int:
    return max(1, round(hours * 10))


def formations_between(formations: tuple[Formation, ...], top: int, bottom: int) -> list[str]:
    """The formation at `top` and every formation whose top lies in (top, bottom]."""
    names = [next((f for f in formations if f.top_m <= top < f.base_m), formations[-1]).name]
    names += [f.name for f in formations if top < f.top_m <= bottom and f.name not in names]
    return names


class WellSim:
    """Plans a well, then writes its daily reports one 24 h period at a time."""

    def __init__(self, well: WellRecord, rng: random.Random, trng: random.Random) -> None:
        self.well = well
        self.spec = well.spec
        self.rng = rng
        self.trng = trng
        self.hole = 0
        self.mw = 0.0
        self.mud = ""
        self.g2_done = False
        self.g3_last_depth: int | None = None   # depth of the last MWD failure, once one has happened
        self.pump_repairs = 0
        self.pill_depth: int | None = None
        self.section_mw = {s.size: round(rng.uniform(*s.mw_range_sg), 2) for s in self.spec.sections}
        self.trip_speed = rng.uniform(330.0, 380.0)
        self.bits = {
            s.size: f"{s.size} roller cone, IADC 115" if i == 0 else f"{s.size} PDC, IADC {trng.choice(BITS)}"
            for i, s in enumerate(self.spec.sections)
        }
        # Shallowest depth of each section where the quoted BHT exceeds the G3 limit.
        self.hot_top = {
            s.size: next((d for d in range(s.top_m, s.base_m) if self.spec.bht_c(d) > G3_BHT_LIMIT_C), None)
            for s in self.spec.sections
        }
        self.queue = self._plan()

    # -- building blocks --------------------------------------------------
    def _fixed(self, sec: SectionPlan, text: str, lo: float, hi: float, *, sets_mw: float | None = None,
               sets_mud: str | None = None, tag: str = "", hint: str = "") -> Op:
        return Op("fixed", sec, text, tenths=tenths(self.rng.uniform(lo, hi)), sets_mw=sets_mw, sets_mud=sets_mud,
                  tag=tag, hint=hint)

    @staticmethod
    def _move(sec: SectionPlan, a: int, b: int, speed: float, text: str, *, tag: str = "", hint: str = "") -> Op:
        return Op("move", sec, text, a=a, b=b, speed=speed, tag=tag, hint=hint)

    def _npt(self, sec: SectionPlan, variant: str, duration: int, depth: int, **params: object) -> Op:
        event = NptEvent(variant, duration, depth, self.spec.formation_at(depth).name, dict(params))
        return Op("npt", sec, tenths=duration, event=event, hint=VARIANTS[variant].hint)

    def _has(self, sec: SectionPlan, name: str, base: int) -> Formation | None:
        """The formation `name` if this section drills into its top."""
        f = next((f for f in self.spec.formations if f.name == name), None)
        return f if f is not None and sec.top_m < f.top_m < base else None

    # -- the plan -----------------------------------------------------------
    def _plan(self) -> list[Op]:
        spec, well = self.spec, self.well
        sections = spec.sections
        ops: list[Op] = []
        if not sections[-1].top_m < well.td_m <= sections[-1].base_m:
            raise RuntimeError(f"{well.name}: TD {well.td_m} m is not in the last hole section")
        for k, sec in enumerate(sections):
            last = k == len(sections) - 1
            base = well.td_m if last else sec.base_m
            mw = self.section_mw[sec.size]
            anchors: list[tuple[int, list[Op]]] = []
            if k == 0:
                if "G4" in spec.patterns and well.rig != G4_RIG:
                    ops.append(self._fixed(sec, "Inspected the mud pumps and changed the fluid ends before spud.",
                                           1.0, 1.0))
                ops.append(self._fixed(sec, f"Made up the {sec.size} BHA and spudded the well.", 1.5, 2.5,
                                       sets_mw=mw, sets_mud=sec.mud_system))
            else:
                prev = sections[k - 1]
                shoe = prev.base_m
                tool = well.mwd_by_section[sec.size]
                bha = f"Made up the {sec.size} BHA."
                if tool != well.mwd_by_section[prev.size]:
                    bha = f"Made up the {sec.size} BHA with a {MWD_VENDOR} {tool} MWD rated to 150 C."
                hint = f"drill out the {prev.casing} shoe and drill ahead in {sec.size} hole."
                ops += [
                    self._fixed(sec, bha, 1.5, 2.5, hint=hint),
                    self._move(sec, 0, shoe - 30, self.trip_speed, "Ran in hole from {a} to {b}.", hint=hint),
                    self._fixed(sec, f"Drilled out the float collar, shoe track and {prev.casing} shoe at "
                                     f"{fmt_m(shoe)} m; displaced the hole to {mw:.2f} sg mud "
                                     f"({sec.mud_system}).", 2.5, 4.0, sets_mw=mw, sets_mud=sec.mud_system,
                                hint=hint),
                ]
                assert sec.fit_range_sg is not None
                fit = round(self.rng.uniform(*sec.fit_range_sg), 2)
                anchors.append((shoe + 3, [self._fixed(
                    sec, f"Performed a FIT to {fit:.2f} sg EMW at {fmt_m(shoe + 3)} m.", 0.8, 1.2)]))
            anchors += self._section_anchors(sec, base)
            ops += self._drill_ops(sec, sec.top_m, base, anchors)
            ops += self._td_sequence(sec, base) if last else self._casing_sequence(k, sec, base)
        return ops

    def _section_anchors(self, sec: SectionPlan, base: int) -> list[tuple[int, list[Op]]]:
        spec, well, rng = self.spec, self.well, self.rng
        out: list[tuple[int, list[Op]]] = []
        salt = self._has(sec, SALT, base)
        if salt is not None:
            at = salt.top_m - 10
            out.append((at, [self._fixed(
                sec, f"Weighted up the mud to {well.salt_mw_sg:.2f} sg at {fmt_m(at)} m before drilling into "
                     f"{SALT}.", 1.0, 2.0, sets_mw=well.salt_mw_sg)]))
            if "G1" in spec.patterns and well.salt_mw_ok:
                for d in range(salt.top_m + 250, salt.base_m, 250):
                    out.append((d, [self._fixed(sec, f"Pumped a saturated brine sweep at {fmt_m(d)} m.", 0.5, 0.5)]))
        carbonate = self._has(sec, CARBONATE, base)
        if carbonate is not None and "G2" in spec.patterns:
            if well.lcm_pretreat:
                pill = rng.randint(carbonate.top_m - 8, carbonate.top_m - 4)
                self.pill_depth = pill
                out.append((pill, [self._fixed(
                    sec, f"Spotted a 30 bbl fibrous LCM pill at {fmt_m(pill)} m, just above the {CARBONATE} top, "
                         f"and cut the flow rate to {PRETREAT_FLOW_GPM} gpm before drilling in.", 0.8, 1.2)]))
            else:
                duration = tenths(rng.uniform(16.0, 28.0))
                out.append((G2_CROSSING_M, [self._npt(sec, "g2_total", duration, G2_CROSSING_M,
                                                      bbl=rng.randint(180, 640))]))
        hot = self.hot_top[sec.size]
        if ("G3" in spec.patterns and sec.size == G3_SECTION and well.mwd_by_section[sec.size] == G3_TOOL
                and hot is not None and rng.random() < G3_FIRST_FAILURE_P):
            # The hotter the hole, the likelier the failure: the depth is drawn
            # with a density that grows linearly from the hot top downwards.
            span = base - 20 - hot
            out.append(self._mwd_failure(sec, hot + round(span * math.sqrt(rng.random()))))
        return out

    def _mwd_failure(self, sec: SectionPlan, depth: int) -> tuple[int, list[Op]]:
        """A round trip to replace the failed module: out of hole and back to bottom."""
        hours = 2 * depth / self.trip_speed + self.rng.uniform(2.5, 5.0)
        return depth, [self._npt(sec, "g3_mwd", tenths(hours), depth, bht=self.spec.bht_c(depth))]

    @staticmethod
    def _drill_ops(sec: SectionPlan, top: int, base: int, anchors: list[tuple[int, list[Op]]]) -> list[Op]:
        ops: list[Op] = []
        cur = top
        for depth, extra in sorted(anchors, key=lambda a: a[0]):
            if not top < depth < base:
                raise RuntimeError(f"planned operation at {depth} m is outside the {sec.size} hole")
            if depth > cur:
                ops.append(Op("drill", sec, a=cur, b=depth))
                cur = depth
            ops += extra
        if base > cur:
            ops.append(Op("drill", sec, a=cur, b=base))
        return ops

    def _casing_sequence(self, k: int, sec: SectionPlan, base: int) -> list[Op]:
        spec, well, rng = self.spec, self.well, self.rng
        nxt = spec.sections[k + 1]
        casing = sec.casing
        trip_text = f"Pulled out of hole from {{a}} to {{b}} to run the {casing}."
        ops = [self._fixed(sec, f"Circulated the hole clean at {fmt_m(base)} m.", 2.0, 3.0,
                           hint=f"pull out of hole and run the {casing}.")]
        salt = self._has(sec, SALT, base)
        if (salt is not None and "G1" in spec.patterns and not well.salt_mw_ok
                and rng.random() < G1_PACKOFF_P):
            ops += self._packoff(sec, base, salt, trip_text)
        else:
            ops.append(self._move(sec, base, 0, self.trip_speed, trip_text, tag="casing_trip_out",
                                  hint=f"run the {casing}."))
        if rng.random() < STANDBY_P:
            ops.append(self._npt(sec, "standby_casing", tenths(rng.uniform(1.5, 6.0)), base))
        ops += [
            self._fixed(sec, "Rigged up the casing running equipment.", 1.5, 2.5, hint=f"run the {casing}."),
            self._move(sec, 0, base, sec.casing_speed_m_per_h, f"Ran the {casing} from {{a}} to {{b}}.",
                       tag="casing_run", hint=f"finish running and cement the {casing}."),
            self._fixed(sec, f"Pumped spacer and cement slurry for the {casing}.", 1.5, 2.5, tag="cement",
                        hint="displace the cement and wait on cement."),
        ]
        if rng.random() < CEMENT_ISSUE_P:
            ops.append(self._npt(sec, "cement", tenths(rng.uniform(2.0, 6.0)), base, casing=casing))
        if k == 0:
            bop = self._fixed(sec, "Nippled up the BOP stack and pressure tested it.", 10.0, 14.0, tag="bop",
                              hint=f"drill out the {casing} shoe and drill ahead in {nxt.size} hole.")
        else:
            bop = self._fixed(sec, "Installed the casing hanger seal assembly and pressure tested the BOP stack.",
                              5.0, 8.0, tag="bop",
                              hint=f"drill out the {casing} shoe and drill ahead in {nxt.size} hole.")
        ops += [
            self._fixed(sec, "Displaced the cement and bumped the plug.", 1.0, 2.0, hint="wait on cement."),
            self._fixed(sec, "Waited on cement.", 8.0, 12.0, hint="wait on cement, then test the BOP."),
            bop,
        ]
        if rng.random() < BOP_FAILURE_P:
            ops.append(self._npt(sec, "bop", tenths(rng.uniform(2.5, 8.0)), base))
        return ops

    def _packoff(self, sec: SectionPlan, base: int, salt: Formation, trip_text: str) -> list[Op]:
        """The G1 trip out: pack-off in the salt, fishing or pumping out, then a cleanout trip."""
        well, rng = self.well, self.rng
        casing = sec.casing
        stuck = rng.randint(max(sec.top_m, salt.top_m) + 20, salt.base_m - 20)
        mw = f"{well.salt_mw_sg:.2f}"
        ops = [self._move(sec, base, stuck, self.trip_speed, trip_text, tag="casing_trip_out",
                          hint=f"run the {casing}.")]
        cleanout = f"run a cleanout trip through {SALT} before the casing."
        if rng.random() < G1_FISHING_P:
            ops.append(self._npt(sec, "g1_packoff_stuck", tenths(rng.uniform(14.0, 24.0)), stuck,
                                 casing=casing, mw=mw))
            # Back off, trip out, make up the fishing string, trip in, jar, trip out
            # with the fish and lay it out: three trips plus the work at surface.
            fishing = 3 * stuck / self.trip_speed + rng.uniform(11.0, 24.0)
            ops.append(self._npt(sec, "g1_fishing", tenths(fishing), stuck))
        else:
            ops += [
                self._npt(sec, "g1_packoff_freed", tenths(rng.uniform(20.0, 40.0)), stuck, casing=casing, mw=mw),
                self._move(sec, stuck, sec.top_m, rng.uniform(150.0, 200.0), "Pumped out of hole from {a} to {b}.",
                           hint=cleanout),
                self._move(sec, sec.top_m, 0, self.trip_speed, "Pulled out of hole from {a} to {b}.", hint=cleanout),
            ]
        ops += [
            self._fixed(sec, "Made up a cleanout assembly.", 1.0, 2.0, hint=cleanout),
            self._move(sec, 0, salt.top_m, self.trip_speed, "Ran in hole from {a} to {b}.", hint=cleanout),
            self._move(sec, salt.top_m, salt.base_m, rng.uniform(120.0, 160.0), f"Reamed {SALT} from {{a}} to {{b}}.",
                       hint=cleanout),
            self._move(sec, salt.base_m, base, self.trip_speed, "Ran in hole from {a} to {b}.",
                       hint=f"circulate, pull out of hole and run the {casing}."),
            self._fixed(sec, f"Circulated the hole clean at {fmt_m(base)} m.", 2.0, 3.0,
                        hint=f"pull out of hole and run the {casing}."),
            self._move(sec, base, 0, self.trip_speed, trip_text, tag="casing_trip_out", hint=f"run the {casing}."),
        ]
        return ops

    def _td_sequence(self, sec: SectionPlan, td: int) -> list[Op]:
        rng = self.rng
        liner = sec.casing
        ops = [
            self._fixed(sec, f"Circulated the hole clean at {fmt_m(td)} m.", 2.0, 3.0,
                        hint="pull out of hole and log the well."),
            self._move(sec, td, 0, self.trip_speed, "Pulled out of hole from {a} to {b} for logging.",
                       tag="td_trip_out", hint="log the well."),
        ]
        if rng.random() < STANDBY_P:
            ops.append(self._npt(sec, "standby_wireline", tenths(rng.uniform(1.5, 6.0)), td))
        ops += [
            self._fixed(sec, f"Rigged up wireline and logged the {sec.size} hole from {fmt_m(td)} m to "
                             f"{fmt_m(sec.top_m)} m.", 12.0, 18.0, tag="log", hint=f"run and cement the {liner}."),
            self._move(sec, 0, td, sec.casing_speed_m_per_h, f"Ran the {liner} on drill pipe from {{a}} to {{b}}.",
                       tag="casing_run", hint=f"finish running and cement the {liner}."),
            self._fixed(sec, f"Pumped spacer and cement slurry for the {liner}.", 1.5, 2.5, tag="cement",
                        hint="displace the cement and set the liner hanger."),
        ]
        if rng.random() < CEMENT_ISSUE_P:
            ops.append(self._npt(sec, "cement", tenths(rng.uniform(2.0, 6.0)), td, casing=liner))
        ops += [
            self._fixed(sec, "Displaced the cement, bumped the plug and set the liner hanger.", 1.5, 2.5,
                        hint="pull out of hole and rig down."),
            self._move(sec, sec.top_m - 100, 0, self.trip_speed,
                       "Pulled out of hole with the liner running tool from {a} to {b}.",
                       hint="rig down and move the rig to the next location."),
            Op("fill", sec, "Rigged down and prepared the rig for the move.",
               hint="rig down and move the rig to the next location."),
        ]
        return ops

    # -- one report ---------------------------------------------------------
    def run(self) -> None:
        """Write daily reports until the plan is done."""
        well = self.well
        n = 0
        while self.queue and not (len(self.queue) == 1 and self.queue[0].kind == "fill"):
            n += 1
            if n > MAX_REPORTS_PER_WELL:
                raise RuntimeError(f"{well.name}: well did not finish in {MAX_REPORTS_PER_WELL} reports")
            factor = self.rng.uniform(0.78, 1.22)
            self._roll(plan_segments(self.queue, self.hole, self.spec, factor))
            segs = plan_segments(self.queue, self.hole, self.spec, factor)
            doc_id = f"DDR-{well.name}-{n:03d}"
            start = self.hole
            acts = run_segments(self.queue, segs, doc_id, self._on_start)
            self.hole = segs[-1].hole_end
            if sum(a.tenths for a in acts) != DAY_TENTHS:
                raise RuntimeError(f"{doc_id} does not cover 24 h")
            well.days.append(self._report(n, doc_id, start, segs, acts))

    def _on_start(self, op: Op) -> None:
        if op.sets_mw is not None:
            self.mw = op.sets_mw
        if op.sets_mud is not None:
            self.mud = op.sets_mud
        if op.event is not None:
            self.well.events.append(op.event)
            op.event.number = len(self.well.events)
            if op.event.variant == "g2_total":
                self.g2_done = True
            if op.event.variant == "g3_mwd":
                self.g3_last_depth = op.event.depth_m
            if op.event.variant == "g4_pump":
                # Worded when it happens, so only a repair that follows another says so.
                op.event.params["repeat"] = " Repeat failure on this pump during the well." if self.pump_repairs else ""
                self.pump_repairs += 1

    def _roll(self, segs: list[Segment]) -> None:
        spec, well, rng = self.spec, self.well, self.rng
        drilled = [s for s in segs if s.op.kind == "drill" and s.hole_end > s.hole_start]
        if drilled:
            a, b = drilled[0].hole_start, drilled[-1].hole_end
            sec = drilled[0].op.section
            if b - a >= 10:
                d = a + round(0.6 * (b - a))
                insert_at_depth(self.queue, d, [self._fixed(
                    sec, f"Circulated bottoms up and took a survey at {fmt_m(d)} m.", 0.5, 1.0)])
            self._roll_drilling(sec, a, b)
        if "G4" in spec.patterns and well.rig == G4_RIG and rng.random() < G4_P:
            t = rng.randrange(DAY_TENTHS)
            duration = tenths(rng.uniform(3.0, 13.0))
            insert_at_time(self.queue, segs, t, lambda sec, depth, _drilling: [
                self._npt(sec, "g4_pump", duration, depth, repeat="")])
        if rng.random() < NOISE_PROBABILITY:
            t = rng.randrange(DAY_TENTHS)
            pick = rng.random()

            def noise(sec: SectionPlan, depth: int, drilling: bool) -> list[Op]:
                table = NOISE_DRILLING if drilling else NOISE_ANY
                variant, lo, hi = table[int(pick * len(table))]
                return [self._npt(sec, variant, tenths(rng.uniform(lo, hi)), depth)]

            insert_at_time(self.queue, segs, t, noise)

    def _roll_drilling(self, sec: SectionPlan, a: int, b: int) -> None:
        spec, well, rng = self.spec, self.well, self.rng
        salt = self._has(sec, SALT, sec.base_m)
        if salt is not None and "G1" in spec.patterns and not well.salt_mw_ok:
            lo, hi = max(a, salt.top_m), min(b, salt.base_m - 1)
            if lo <= hi and rng.random() < G1_CREEP_P:
                d = rng.randint(lo, hi)
                insert_at_depth(self.queue, d, [self._npt(sec, "g1_creep", tenths(rng.uniform(2.0, 7.0)), d,
                                                          overpull=rng.randint(18, 32),
                                                          mw=f"{well.salt_mw_sg:.2f}")])
        carbonate = self._has(sec, CARBONATE, sec.base_m)
        if carbonate is not None and "G2" in spec.patterns and not well.lcm_pretreat and self.g2_done:
            lo, hi = max(a, G2_CROSSING_M + 1), min(b, G2_PARTIAL_BASE_M - 1)
            if lo <= hi and rng.random() < G2_PARTIAL_P:
                d = rng.randint(lo, hi)
                insert_at_depth(self.queue, d, [self._npt(sec, "g2_partial", tenths(rng.uniform(6.0, 14.0)), d,
                                                          rate=rng.randint(15, 40))])
        if self.g3_last_depth is not None and sec.size == G3_SECTION:
            # A replacement module drills at least 30 m before it can fail again.
            lo, hi = max(a, self.g3_last_depth + 30), min(b, sec.base_m - 20)
            if lo <= hi and rng.random() < G3_REPEAT_P:
                insert_at_depth(self.queue, *self._mwd_failure(sec, rng.randint(lo, hi)))

    def _report(self, n: int, doc_id: str, start: int, segs: list[Segment], acts: list[Activity]) -> DayReport:
        spec, well, t = self.spec, self.well, self.trng
        end = self.hole
        sec = segs[-1].op.section
        drilled = [s for s in segs if s.op.kind == "drill" and s.hole_end > s.hole_start]
        entries = [a.entry for a in acts if a.entry is not None]
        formations = formations_between(spec.formations, start, end)
        formations += [e.formation for e in entries if e.formation not in formations]
        cosmetics: dict[str, object] = {"pv": t.randint(16, 32), "yp": t.randint(12, 26), "bit": self.bits[sec.size]}
        remarks = [f"Casing programme for this section: {sec.casing}."]
        if drilled:
            wob, rpm, flow = DRILLING_PARAMETERS[sec.size]
            cosmetics.update(ecd=round(self.mw + t.uniform(0.05, 0.10), 2), wob=t.randint(*wob),
                             rpm=t.randint(*rpm), flow=t.randint(*flow))
            carbonate = self._has(sec, CARBONATE, sec.base_m)
            if carbonate is not None and self.pill_depth is not None:
                a, b = drilled[0].hole_start, drilled[-1].hole_end
                if a < carbonate.top_m + PRETREAT_INTERVAL_M and b > self.pill_depth:
                    cosmetics["flow"] = PRETREAT_FLOW_GPM
                if a < carbonate.top_m + PRETREAT_INTERVAL_M and b > carbonate.top_m:
                    remarks.append(f"Seepage losses of {t.randint(2, 4)} bbl/h while drilling {CARBONATE} with "
                                   f"LCM in the system.")
        remarks.append(f"Next 24 h: {self._outlook()}")
        return DayReport(
            doc_id=doc_id, report_no=n, section=sec, depth_start=start, depth_end=end,
            formation_end=spec.formation_at(end), mud_system=self.mud, mud_weight_sg=self.mw, bht_c=spec.bht_c(end),
            mwd=well.mwd_by_section[sec.size], drilled=bool(drilled), entries=entries, activities=acts,
            remarks=remarks, cosmetics=cosmetics, formations=formations,
        )

    def _outlook(self) -> str:
        head = self.queue[0] if self.queue else None
        if head is None or head.kind == "fill":
            return "rig down and move the rig to the next location."
        if head.hint:
            return head.hint
        return f"continue drilling {head.section.size} hole in {self.spec.formation_at(self.hole).name}."

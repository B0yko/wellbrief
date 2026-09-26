"""What each kind of non-productive time event says, and what an incident report does about it.

An event can run past 06:00 into the next daily report. Each report then
carries one NPT DETAIL entry for its share of the hours, so each variant has
four texts: `whole` for an event that fits in one report, `first` for the
share in the report where it starts, `middle` for a report it fills
completely and `last` for the report where it ends. Only `whole` may quote a
duration, because only there is the event's duration the entry's hours.
Every continuation text starts with "Continued", so a reader can tell the
start of an event from its continuation.

The texts are format strings. `d` is the entry depth, `h` the hours, and the
other names come from the event's parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fields import CARBONATE, MWD_VENDOR, SALT


@dataclass(frozen=True)
class Variant:
    code: str
    pattern: str | None
    kind: str        # "drilling": at hole depth; "trip_out": above hole depth, on a trip out
    whole: str
    first: str
    middle: str
    last: str
    hint: str        # the "Next 24 h" remark when the event is still running at 06:00


def _v(code: str, pattern: str | None, whole: str, first: str | None, middle: str, last: str | None,
       hint: str, kind: str = "drilling") -> Variant:
    return Variant(code, pattern, kind, whole, first or whole, middle, last or middle, hint)


VARIANTS: dict[str, Variant] = {
    # G1: salt creep at low mud weight, a pack-off on the trip out, sometimes fishing.
    "g1_creep": _v(
        "WELLBORE_INSTABILITY", "G1",
        "Tight hole and overpull up to {overpull} t on connections at {d} m; salt creep at {mw} sg. "
        "Reamed the interval over {h} h.",
        "Tight hole and overpull up to {overpull} t on connections at {d} m; salt creep at {mw} sg. "
        "Reaming the interval.",
        "Continued reaming the tight salt interval at {d} m.",
        "Continued reaming the tight salt interval at {d} m until the hole was free.",
        "finish reaming the tight salt interval, then drill ahead."),
    "g1_packoff_freed": _v(
        "STUCK_PIPE", "G1",
        "String packed off at {d} m while pulling out of hole for the {casing}. Salt creep at {mw} sg "
        f"across {SALT}. Worked pipe and jarred, spotted a fresh water pill across the salt and jarred free "
        "after {h} h.",
        "String packed off at {d} m while pulling out of hole for the {casing}. Salt creep at {mw} sg "
        f"across {SALT}. Working pipe and jarring.",
        "Continued working the stuck string at {d} m.",
        "Continued working the stuck string at {d} m; spotted a fresh water pill across the salt and jarred "
        "free.",
        "work the stuck string free.", kind="trip_out"),
    "g1_packoff_stuck": _v(
        "STUCK_PIPE", "G1",
        "String packed off at {d} m while pulling out of hole for the {casing}. Salt creep at {mw} sg "
        f"across {SALT}. Worked pipe, jarred and spotted a fresh water pill for {{h}} h without freeing "
        "the string.",
        "String packed off at {d} m while pulling out of hole for the {casing}. Salt creep at {mw} sg "
        f"across {SALT}. Working pipe and jarring.",
        "Continued working the stuck string at {d} m.",
        "Continued working and jarring the stuck string at {d} m without freeing it.",
        "work the stuck string; prepare to back off and fish.", kind="trip_out"),
    "g1_fishing": _v(
        "FISHING", "G1",
        "Backed off above the jars and ran a fishing assembly to recover the BHA stuck at {d} m. "
        "Jarred the fish free, pulled out and laid out the BHA.",
        "Backed off above the jars and started fishing for the BHA stuck at {d} m.",
        "Continued fishing for the BHA stuck at {d} m.",
        "Continued fishing for the BHA stuck at {d} m; jarred the fish free, pulled out and laid out "
        "the BHA.",
        "continue fishing.", kind="trip_out"),
    # G2: total losses on entering the carbonate, partial losses below it.
    "g2_total": _v(
        "LOST_CIRCULATION", "G2",
        f"Total losses on entering {CARBONATE} at {{d}} m. Returns lost, no fluid to surface; lost "
        "{bbl} bbl mud. Pumped LCM pills, set a cement plug, waited on cement and drilled out the plug.",
        f"Total losses on entering {CARBONATE} at {{d}} m. Returns lost, no fluid to surface; lost "
        "{bbl} bbl mud. Pumping LCM pills.",
        "Continued curing total losses at {d} m with LCM pills and a cement plug.",
        "Continued curing total losses at {d} m; waited on cement, drilled out the plug and regained "
        "returns.",
        "continue curing the losses."),
    "g2_partial": _v(
        "LOST_CIRCULATION", "G2",
        "Partial losses of {rate} bbl/h below the carbonate top at {d} m. Pumped LCM sweeps until "
        "returns were regained.",
        "Partial losses of {rate} bbl/h below the carbonate top at {d} m. Pumping LCM sweeps.",
        "Continued pumping LCM sweeps for partial losses at {d} m.",
        "Continued pumping LCM sweeps for partial losses at {d} m until returns were regained.",
        "cure the partial losses, then drill ahead."),
    # G3: the PJ-3 MWD fails above its temperature rating.
    "g3_mwd": _v(
        "DOWNHOLE_TOOL_FAILURE", "G3",
        f"MWD signal lost at {{d}} m, BHT {{bht}} C. Pulled out of hole, replaced the {MWD_VENDOR} PJ-3 "
        "directional module and ran back to bottom. Tool memory shows the module above 115 C before "
        "the failure.",
        f"MWD signal lost at {{d}} m, BHT {{bht}} C. Pulling out of hole to replace the {MWD_VENDOR} "
        "PJ-3 directional module.",
        "Continued the round trip for the PJ-3 MWD module that failed at {d} m, BHT {bht} C.",
        "Continued the round trip: replaced the PJ-3 MWD module that failed at {d} m, BHT {bht} C, "
        "and ran back to bottom.",
        "finish the MWD round trip, then drill ahead."),
    # G4: the mud pump fluid end on one rig.
    "g4_pump": _v(
        "RIG_REPAIR", "G4",
        "Mud pump #2 fluid end module changed out after a washed valve seat.{repeat}",
        "Mud pump #2 washed a valve seat; changing out the fluid end module.{repeat}",
        "Continued changing out the mud pump #2 fluid end module.",
        "Continued and completed the mud pump #2 fluid end change-out.",
        "finish the mud pump repair."),
    # Background NPT with no planted cause.
    "weather": _v(
        "WAIT_ON_WEATHER", None,
        "Storm warning, operations suspended. Wind gusting above 40 knots.", None,
        "Continued waiting on weather; wind above 40 knots.",
        "Continued waiting on weather until the wind dropped below 40 knots.",
        "resume operations when the weather allows."),
    "materials": _v(
        "WAIT_ON_MATERIALS", None,
        "Waiting on delivery of barite from the shore base.", None,
        "Continued waiting on barite from the shore base.",
        "Continued waiting on barite until the delivery from the shore base arrived.",
        "resume operations once the barite arrives."),
    "drawworks": _v(
        "RIG_REPAIR", None,
        "Repaired drawworks brake linkage.",
        "Drawworks brake linkage failed; repairing.",
        "Continued the drawworks brake linkage repair.",
        "Continued and completed the drawworks brake linkage repair.",
        "finish the drawworks repair."),
    "hse": _v(
        "HSE_STOP", None,
        "Stop-work called on rig floor, lifting plan reviewed before restart.", None,
        "Continued the stop-work review of the lifting plan.",
        "Continued the stop-work review; lifting plan approved and work restarted.",
        "restart once the lifting plan is approved."),
    "hole_cleaning": _v(
        "HOLE_CLEANING", None,
        "Circulated and back-reamed at {d} m to clear a cuttings bed.", None,
        "Continued back-reaming at {d} m to clear the cuttings bed.",
        "Continued back-reaming at {d} m until the cuttings bed was cleared.",
        "finish clearing the cuttings bed, then drill ahead."),
    "bop": _v(
        "BOP_TEST_FAILURE", None,
        "Annular preventer failed the low pressure test; replaced the element and retested.",
        "Annular preventer failed the low pressure test; replacing the element.",
        "Continued replacing the annular preventer element.",
        "Continued replacing the annular preventer element and retested it.",
        "finish the BOP repair and retest."),
    "cement": _v(
        "CEMENT_ISSUE", None,
        "Cement head seal leaked while cementing the {casing}; replaced the seal before displacement.",
        "Cement head seal leaked while cementing the {casing}; replacing the seal.",
        "Continued replacing the cement head seal on the {casing} job.",
        "Continued replacing the cement head seal on the {casing} job; ready to displace.",
        "finish the cement head repair and displace the cement."),
    "standby_casing": _v(
        "THIRD_PARTY_STANDBY", None,
        "Casing running crew standby, crew mobilising from the shore base.", None,
        "Continued standby for the casing running crew.",
        "Continued standby until the casing running crew arrived.",
        "run casing once the crew arrives."),
    "standby_wireline": _v(
        "THIRD_PARTY_STANDBY", None,
        "Wireline crew standby, unit mobilising from the shore base.", None,
        "Continued standby for the wireline unit.",
        "Continued standby until the wireline unit arrived.",
        "log the well once the wireline unit arrives."),
}

# Background events that can interrupt any operation, and the ones that only
# make sense while drilling: (variant, min hours, max hours).
NOISE_ANY: tuple[tuple[str, float, float], ...] = (
    ("weather", 2.0, 9.0),
    ("materials", 2.0, 11.0),
    ("drawworks", 1.5, 7.0),
    ("hse", 1.0, 3.5),
)
NOISE_DRILLING: tuple[tuple[str, float, float], ...] = (*NOISE_ANY, ("hole_cleaning", 2.0, 8.0))
NOISE_PROBABILITY = 0.18


# ---------------------------------------------------------------------------
# Incident reports
# ---------------------------------------------------------------------------

ROOT_CAUSES = {
    "STUCK_PIPE": ("Mud weight below the value required to hold the salt in gauge, combined with an "
                   "extended open hole exposure time."),
    "FISHING": "The string could not be worked free after the pack-off in the creeping salt interval.",
    "LOST_CIRCULATION": ("The carbonate top was entered at full flow rate without hole conditioning or "
                         "a pre-emptive LCM pill."),
    "DOWNHOLE_TOOL_FAILURE": ("Directional tool operated above its demonstrated continuous temperature "
                              "rating for an extended period."),
    "RIG_REPAIR": ("Fluid end module on mud pump #2 reached end of service life and was not replaced "
                   "during the previous maintenance window."),
    "WELLBORE_INSTABILITY": "Salt creep at a mud weight below the value needed to hold Keldra Salt in gauge.",
}
DEFAULT_ROOT_CAUSE = "Equipment or process deviation, see sequence of events."

# Corrective actions per NPT code: (text, label, patterns). The first action
# of each code answers its root cause and is on every incident report of that
# code; one or two of the others are added. Every code has its own list, so
# an action never appears under two codes.
CORRECTIVE_ACTIONS: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    "STUCK_PIPE": (
        ("Raise mud weight to at least 1.42 sg before drilling into Keldra Salt.", "practice", ("G1",)),
        ("Pump out of hole through the salt interval rather than pulling on elevators.", "practice", ("G1",)),
        ("Run a caliper across the salt before the trip out for casing.", "practice", ("G1",)),
    ),
    "FISHING": (
        ("Place the jars so that the string can be backed off above the stuck point in the salt.",
         "practice", ("G1",)),
        ("Keep a fishing assembly for the 17 1/2\" BHA on location while drilling the salt.", "practice", ("G1",)),
    ),
    "WELLBORE_INSTABILITY": (
        ("Weight up to at least 1.42 sg across Keldra Salt and ream tight spots before connections.",
         "practice", ("G1",)),
        ("Record overpull on every connection in the salt and review the trend daily.", "practice", ("G1",)),
    ),
    "LOST_CIRCULATION": (
        (("Spot a fibrous LCM pill just above the Vessra Carbonate top and cut flow rate below 450 gpm "
          "before drilling in."), "practice", ("G2",)),
        ("Keep 200 bbl of pre-mixed LCM pill on location while drilling the 8 1/2\" section.",
         "practice", ("G2",)),
        ("Monitor pit volumes for losses every 15 minutes while drilling the first 60 m of Vessra Carbonate.",
         "practice", ("G2",)),
    ),
    "DOWNHOLE_TOOL_FAILURE": (
        ("Run MWD tools rated to at least 150 C where BHT exceeds 118 C.", "practice", ("G3",)),
        ("Replace Parvane Downhole PJ-3 MWD modules with PJ-5 for the 12 1/4\" section.", "practice", ("G3",)),
        ("Log circulating temperature at every connection below 2,300 m to track the MWD temperature.",
         "practice", ("G3",)),
    ),
    "RIG_REPAIR": (
        ("Inspect and change mud pump fluid ends between wells.", "practice", ("G4",)),
        ("Keep a spare fluid end module for each mud pump on location.", "practice", ("G4",)),
    ),
    "HOLE_CLEANING": (
        ("Pump a high-viscosity sweep to lift cuttings before every trip out of hole.", "practice", ()),
        ("Keep annular velocity above the cuttings transport minimum in the upper hole.", "practice", ()),
    ),
    "CEMENT_ISSUE": (
        ("Pressure test the cement head before the cement job.", "practice", ()),
        ("Keep a spare cement head seal kit on location.", "practice", ()),
    ),
    "BOP_TEST_FAILURE": (
        ("Replace annular preventer elements at the interval given by the equipment vendor.", "practice", ()),
        ("Keep a spare annular preventer element on location.", "practice", ()),
    ),
    "WAIT_ON_MATERIALS": (
        ("Hold two days of barite stock on location.", "practice", ()),
        ("Confirm barite supply boat schedules with the shore base each morning.", "practice", ()),
    ),
    "WAIT_ON_WEATHER": (
        ("Plan critical operations around the weather forecast window.", "practice", ()),
        ("The weather stop was outside the crew's control.", "neutral", ()),
    ),
    "THIRD_PARTY_STANDBY": (
        ("Call out third-party crews 24 h before they are needed.", "practice", ()),
        ("The standby time of the third-party crew is recorded against the service contract.", "neutral", ()),
    ),
    "HSE_STOP": (
        ("Review lifting plans at the pre-tour meeting.", "practice", ()),
        ("The stop-work call was made correctly and no one was hurt.", "neutral", ()),
    ),
}

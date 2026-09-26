"""Synthetic well-file corpus.

All data here is synthetic and every name is fictional: the operator, fields,
wells, rigs, formations and vendors are invented. The generated field history
is shaped like a real one: each well is drilled as a sequence of operations
(drilling, trips, casing and cement jobs, logging) that takes the time it
takes, rates of penetration follow formation, non-productive time clusters
around physical causes, and the paperwork follows the layout of a daily
drilling report, an end of well report and an incident report. Every
document ends with a footer line that says it is synthetic.

Four causal patterns are planted in the data. Nothing in the retrieval or
analytics code knows about them; they have to be rediscovered from the
documents, which is the task the tool is built for.

  G1  Orrindale, 17 1/2" through Keldra Salt. Wells run below 1.38 sg see
      tight hole from salt creep and, once per well, pack off on the trip
      out for the intermediate casing, sometimes followed by fishing.
  G2  Vessra South, 8 1/2" entering Vessra Carbonate near 2,660 m. Total
      losses, once per well, unless an LCM pill is spotted and the flow rate
      cut before the carbonate top; partial losses may follow.
  G3  Both fields, 12 1/4" where BHT exceeds 118 C. The PJ-3 MWD generation
      fails at temperature; PJ-5 does not.
  G4  Rig Vessra-3 carries repeated mud pump fluid end repairs.

The corpus is deterministic for a seed: the same seed and scale give
byte-identical files and the same `manifest_hash`. The generator also
writes a ground-truth sidecar next to the documents (see `truth.py`).

Modules: `fields` (geometry, names, mud programme), `events` (what each NPT
event and incident report says), `timeline` (operations cut into 24 h
reports), `simulate` (the plan and the daily events of one well), `writeup`
(lessons, recommendations and incident reports), `generator` (the campaign),
`render` (records to lines), `writers` (lines to files, one writer per
format, and the manifest hash), `truth` (the sidecar).
"""

from __future__ import annotations

from .fields import FIELDS, OPERATOR, ORRINDALE, VESSRA_SOUTH
from .generator import MAX_SCALE, SEED, build_corpus
from .records import Corpus, RenderedDocument
from .writers import FORMATS, WRITERS, OutputDirError, manifest_hash, register_writer, write_corpus

__all__ = [
    "FIELDS", "FORMATS", "MAX_SCALE", "OPERATOR", "ORRINDALE", "SEED", "VESSRA_SOUTH", "WRITERS",
    "Corpus", "OutputDirError", "RenderedDocument", "build_corpus", "manifest_hash",
    "register_writer", "write_corpus",
]

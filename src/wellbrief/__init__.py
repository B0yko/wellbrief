"""wellbrief - offline retrieval and NPT analytics over drilling well files.

Runs without network access and without a model download. Point it at a
folder of plain-text daily drilling reports, end of well reports and
incident reports in the supported template, and it answers questions with
citations, totals the non-productive time, finds what repeats across wells,
and builds an offset-well risk register for a planned well.
"""

__version__ = "0.1.0"

from .models import Answer, Citation, Document, NptEvent, Risk, RiskBrief, Well

__all__ = [
    "Answer", "Citation", "Document", "NptEvent", "Risk", "RiskBrief", "Well", "__version__",
]

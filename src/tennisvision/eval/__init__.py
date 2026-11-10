"""Banc d'évaluation : métriques par brique et par statistique, rapport reproductible."""

from tennisvision.eval.banc import (
    charger_verites_amateur,
    ecrire_rapport,
    evaluer,
    plis_leave_one_out,
)

__all__ = ["charger_verites_amateur", "ecrire_rapport", "evaluer", "plis_leave_one_out"]

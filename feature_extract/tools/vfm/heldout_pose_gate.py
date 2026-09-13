"""Evidence gates over query tokens unused by regional proposal generation.

This is measurement separation, not statistical independence: encoder and map
are shared. Absence of support and contrary support are handled differently.
"""


def heldout_gate(base_score, candidate_score, mode):
    if mode == 'veto':
        # Require positive baseline evidence before calling the new mode worse.
        return not (base_score[0] >= 6 and base_score[0] > candidate_score[0])
    if mode == 'corroborate':
        return candidate_score[0] >= 6 and tuple(candidate_score) > tuple(base_score)
    raise ValueError('unknown heldout gate')

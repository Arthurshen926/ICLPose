"""Separate candidate eligibility from ranking, with an explicit evidence rescue.

The two token domains share an encoder/map; agreement is not independence or a
calibrated probability. The rescue is an experimental rule, not a safety claim.
"""
from feature_extract.tools.vfm.heldout_pose_gate import heldout_gate


def select_candidate(probabilities, separated, proposal_scores, bank_scores,
                     threshold, policy='eligible'):
    if policy not in ('eligible', 'dual_support'):
        raise ValueError('unknown selection policy')
    n=len(probabilities)
    if not (len(separated)==len(proposal_scores)==len(bank_scores)==n and n):
        raise ValueError('candidate arrays differ')
    eligible=[j for j in range(1,n) if separated[j]
              and probabilities[j]>=threshold
              and heldout_gate(bank_scores[0],bank_scores[j],'corroborate')]
    if eligible:
        return max(eligible,key=lambda j:probabilities[j]),'relative_and_bank'
    if policy=='dual_support':
        rescue=[j for j in range(1,n) if separated[j]
                and proposal_scores[j][0]>=6
                and tuple(proposal_scores[j])>tuple(proposal_scores[0])
                and heldout_gate(bank_scores[0],bank_scores[j],'corroborate')]
        if rescue:
            # Rank on generator evidence; use the reserved bank only as a gate.
            return max(rescue,key=lambda j:tuple(proposal_scores[j])),'dual_support_rescue'
    return 0,'retain_reference'

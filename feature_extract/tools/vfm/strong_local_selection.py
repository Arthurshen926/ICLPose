"""Forward local-update selection on a fixed, authenticated strong endpoint."""
import numpy as np


def choose_local(rows, scores, threshold, enabled=True):
    if not enabled:return -1
    if len(rows)!=len(scores):raise ValueError('score/pair length differs')
    eligible=[i for i,x in enumerate(rows) if x['valid_local_pair'] and x['support']>=6 and np.isfinite(scores[i]) and scores[i]>=threshold]
    return max(eligible,key=lambda i:scores[i]) if eligible else -1


def threshold_harm(base_error, candidate_error):
    return any(base_error[0]<=t and base_error[1]<=a
               and not(candidate_error[0]<=t and candidate_error[1]<=a)
               for t,a in [(.1,1),(.25,2),(.5,5)])


def calibrate_forward(groups, probabilities, minimum_queries=5, minimum_score=.5):
    """Only actual forward decisions qualify; reversed training pairs do not."""
    candidates=sorted({minimum_score}|{float(v) for vs in probabilities for v in vs if np.isfinite(v) and v>=minimum_score})
    for threshold in candidates:
        decisions=[]
        for rows,scores in zip(groups,probabilities):
            j=choose_local(rows,scores,threshold)
            if j>=0:decisions.append(rows[j])
        if len({x['name'] for x in decisions})>=minimum_queries and not any(threshold_harm(x['base_error'],x['candidate_error']) for x in decisions):
            return threshold,True
    return 1.,False


def choose_multiscale_local(rows):
    """Both coarse and fine mean cosine gains must be strictly positive."""
    eligible=[i for i,x in enumerate(rows) if x['valid_local_pair'] and x['support']>=6
              and np.isfinite(x['features']).all()
              and x['features'][0]>0 and x['features'][6]>0]
    return max(eligible,key=lambda i:rows[i]['features'][6]) if eligible else -1

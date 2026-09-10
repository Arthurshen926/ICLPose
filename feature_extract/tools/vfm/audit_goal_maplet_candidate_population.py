"""Measure retrieved candidate coverage using source-image/token observations."""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def population(sources,tokens,labels):
    _,group=np.unique(np.c_[sources,tokens],axis=0,return_inverse=True)
    size=int(group.max())+1 if len(group) else 0
    positive=np.zeros(size,bool);ambiguous=np.zeros(size,bool)
    np.logical_or.at(positive,group,np.asarray(labels)==1)
    np.logical_or.at(ambiguous,group,np.asarray(labels)==-1)
    return {'independent_tokens':size,'with_positive_evidence':int(positive.sum()),
            'without_positive_but_with_ambiguous_evidence':int((~positive&ambiguous).sum()),
            'only_negative_evidence':int((~positive&~ambiguous).sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    c=np.load(a.candidates,allow_pickle=False);l=np.load(a.labels,allow_pickle=False)
    if str(l['frozen_candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('label lineage differs')
    report={'unit':'one source image and query token, not one mapping plane-observation row',
            'candidate_sha256':file_sha256(a.candidates),'limitations':'only tokens present in MNN candidate inventory; not every observed query token'}
    for name,mask in [('before_homography',np.ones(len(c['query_rows']),bool)),('after_homography',c['homography_keep'])]:
        report[name]=population(c['source_image'][mask],c['query_token'][mask],l['labels'][mask])
        report[name]['distinct_observation_rows']=int(len(np.unique(c['query_rows'][mask])))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

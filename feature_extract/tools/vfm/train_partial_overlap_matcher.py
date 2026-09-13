"""Train overlap and conditional identity; calibrate on mapping seq7 only."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from scipy.optimize import minimize
from scipy.special import expit
from feature_extract.tools.vfm.partial_overlap_matcher import PartialOverlapMatcher,masked_losses
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def load_example(path):
    with np.load(path) as f:return {k:torch.from_numpy(f[k].astype(np.float32) if k in ['query','map','edges','similarity','target'] else f[k].astype(bool)) for k in ['query','map','edges','similarity','target','positive','known']}


def batch(examples,device):
    maximum=max(len(x['target']) for x in examples);out={}
    for k in examples[0]:
        arrays=[]
        for e in examples:
            v=e[k];n=len(v)
            if n<maximum:
                shape=(maximum,maximum) if k=='edges' else (maximum,*v.shape[1:]);padded=torch.full(shape,-1 if k=='target' else 0,dtype=v.dtype)
                if k=='edges':padded[:n,:n]=v
                else:padded[:n]=v
                v=padded
            arrays.append(v)
        out[k]=torch.stack(arrays).to(device)
    return out


def evaluate(model,data,device):
    model.eval();records=[]
    with torch.no_grad():
        for e,r in data:
            x=batch([e],device);overlap,identity=model(x['query'],x['map'],x['edges'],x['similarity']);pred=identity.argmax(-1);chosen=x['positive'].gather(-1,pred[...,None]).squeeze(-1)
            records.append(dict(route=r['route'],name=r['name'],target=e['target'].numpy().tolist(),overlap_logits=overlap[0].cpu().tolist(),identity_correct=chosen[0].cpu().tolist(),identity_available=e['positive'].any(-1).tolist(),coarse_identity_correct=e['positive'][torch.arange(len(e['target'])),e['similarity'][...,0].argmax(-1)].tolist()))
    return records


def report(records,calibration):
    y=np.concatenate([r['target'] for r in records]);z=np.concatenate([r['overlap_logits'] for r in records]);valid=y>=0;y=y[valid];prob=expit(calibration[0]*z[valid]+calibration[1]);pred=prob>=.5
    available=np.concatenate([r['identity_available'] for r in records]);correct=np.concatenate([r['identity_correct'] for r in records]);coarse=np.concatenate([r['coarse_identity_correct'] for r in records])
    return dict(known_tokens=len(y),positive_fraction=float(y.mean()),overlap_precision=float(y[pred].mean()) if pred.any() else 0.,overlap_recall=float(pred[y==1].mean()) if (y==1).any() else 0.,overlap_brier=float(np.mean((prob-y)**2)),conditional_identity_top1=float(correct[available].mean()),coarse_identity_top1=float(coarse[available].mean()),identity_available_tokens=int(available.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=402);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=12);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed);torch.backends.cudnn.benchmark=False;manifest=json.load(open(a.data/'manifest.json'));train=[];validation=[]
    for r in manifest['records']:
        assert r['route'] not in ['seq10','seq13'];e=load_example(a.data/r['path']);(train if r['split']=='train' else validation).append((e,r))
    known=np.concatenate([e['target'].numpy() for e,r in train]);weight=float((known==0).sum()/max(1,(known==1).sum()));model=PartialOverlapMatcher().to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3);history=[];rng=np.random.default_rng(a.seed);started=time.perf_counter()
    for epoch in range(a.epochs):
        model.train();order=rng.permutation(len(train));losses=[]
        for start in range(0,len(order),4):
            x=batch([train[j][0] for j in order[start:start+4]],a.device)
            # Missing rectangular evidence augmentation: ignore its targets,
            # never relabel a hidden surface as a visible positive or negative.
            if rng.random()<.5:
                xy=x['query'][...,-8:-6];center=torch.tensor(rng.uniform(-.6,.6,2),device=a.device);half=torch.tensor(rng.uniform(.15,.5,2),device=a.device);hidden=((xy-center).abs()<half).all(-1);x['query']=x['query'].masked_fill(hidden[...,None],0);x['target']=x['target'].masked_fill(hidden,-1);x['positive']=x['positive']&~hidden[...,None];x['known']=x['known']&~hidden[...,None];x['edges']=x['edges']*(~hidden)[:,:,None]*(~hidden)[:,None,:]
            overlap,identity=model(x['query'],x['map'],x['edges'],x['similarity']);_,_,identity_loss=masked_losses(overlap,identity,x['target'],x['positive'],x['known']);valid=x['target']>=0
            overlap_loss=torch.nn.functional.binary_cross_entropy_with_logits(overlap[valid],x['target'][valid],pos_weight=torch.tensor(weight,device=a.device)) if valid.any() else overlap.sum()*0
            loss=overlap_loss+identity_loss
            if not torch.isfinite(loss):raise ValueError('nonfinite training objective')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append([float(loss.detach()),float(overlap_loss.detach()),float(identity_loss.detach())])
        history.append(dict(epoch=epoch+1,losses=np.mean(losses,axis=0).tolist()));print(history[-1],flush=True)
    # Fixed epoch schedule; no model/hyperparameter selection on seq8/seq11.
    results=evaluate(model,validation,a.device);cal=[r for r in results if r['route']=='seq7'];y=np.concatenate([r['target'] for r in cal]);z=np.concatenate([r['overlap_logits'] for r in cal]);known=y>=0;y=y[known];z=z[known]
    def objective(v):
        logits=np.exp(v[0])*z+v[1];return float(np.mean(np.logaddexp(0,logits)-y*logits))
    fit=minimize(objective,[0.,0.],method='L-BFGS-B',bounds=[(-3,3),(-10,10)]);calibration=[float(np.exp(fit.x[0])),float(fit.x[1])]
    meta=dict(seed=a.seed,epochs=a.epochs,training_routes=manifest['training_routes'],calibration_routes=['seq7'],confirmation_routes=['seq8','seq11'],training_pairs=len(train),validation_pairs=len(validation),train_manifest_sha256=file_sha256(a.data/'manifest.json'),source_sha256={str(p):file_sha256(p) for p in [Path(__file__),Path(__file__).with_name('partial_overlap_matcher.py')]},query_test_routes_opened=False,coordinate_head_trained=False,map_members_trained=False,overlap_calibration=calibration,calibration_optimizer_success=bool(fit.success),seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),class_weight=weight)
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},metadata=meta),a.output/'matcher.pt');(a.output/'metadata.json').write_text(json.dumps(meta,indent=2));(a.output/'history.json').write_text(json.dumps(history,indent=2));(a.output/'validation_predictions.json').write_text(json.dumps(results));summaries={route:report([r for r in results if r['route']==route],calibration) for route in ['seq7','seq8','seq11']};(a.output/'validation.json').write_text(json.dumps(summaries,indent=2));print(summaries,flush=True)
if __name__=='__main__':main()

"""Reproducible USGS annual count forecasts. Run: python analysis.py.
No network access. Frozen 2020 output is write-once; existing output must match.
"""
import json, hashlib, time, math
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, expit, logsumexp
from scipy.stats import poisson, nbinom

ROOT=Path(__file__).resolve().parent
SEED=20261107
MODELS=['iid_poisson','iid_nb2','iid_poisson_mix','poisson_ar','poisson_ar_mix']
BOUNDS={'iid_nb2':[(-8,8),(1e-8,10)],'iid_poisson_mix':[(-8,8),(-8,8),(-12,12)],'poisson_ar':[(-8,8),(-.95,.95)],'poisson_ar_mix':[(-8,8),(-.95,.95),(-8,8),(-.95,.95),(-12,12)]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_counts():
    obj=json.loads((ROOT/'usgs_1918_2020_snapshot.json').read_text(encoding='utf-8-sig'))
    counts=np.zeros(103,dtype=int); ids=[]
    for f in obj['geojson']['features']:
        p=f['properties']; year=(datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(milliseconds=p['time'])).year
        assert 1918<=year<=2020 and p['mag']>=7 and p['type']=='earthquake'
        ids.append(f['id']);counts[year-1918]+=1
    assert len(ids)==len(set(ids))==1275 and counts.sum()==1275
    return counts

def objective(p,model,y,x):
    c=gammaln(y+1)
    if model=='poisson_ar':
        eta=p[0]+p[1]*x;m=np.exp(eta); e=m-y
        return float(np.sum(m-y*eta+c)),np.array([e.sum(),e@x])
    if model=='iid_nb2':
        m=np.exp(p[0]);a=p[1];r=1/a
        return float(-nbinom.logpmf(y,r,r/(r+m)).sum())
    ar=model=='poisson_ar_mix'
    if ar: e1=p[0]+p[1]*x;e2=p[2]+p[3]*x;t=p[4]
    else:e1=np.full(len(y),p[0]);e2=np.full(len(y),p[1]);t=p[2]
    m1=np.exp(e1);m2=np.exp(e2);w=expit(t)
    l1=-np.logaddexp(0,-t)+y*e1-m1-c;l2=-np.logaddexp(0,t)+y*e2-m2-c
    l= np.logaddexp(l1,l2);z=np.exp(l1-l);d1=z*(m1-y);d2=(1-z)*(m2-y)
    g=[d1.sum(),d1@x,d2.sum(),d2@x,(w-z).sum()] if ar else [d1.sum(),d2.sum(),(w-z).sum()]
    return float(-l.sum()),np.array(g)

def fit_all(data,origin):
    y=np.asarray(data[1:],float);x=np.log1p(data[:-1]);n=len(y);mean=y.mean();a=np.log(mean)
    base=float(-poisson.logpmf(y,mean).sum())
    fits={'iid_poisson':{'params':[a],'objective':base,'selected_source':'closed_form','starts':[{'source':'closed_form','objective':base,'success':True,'iterations':0,'params':[a],'boundary':[]} ]}}
    rng=np.random.default_rng(SEED+origin)
    for model in MODELS[1:]:
        starts=[];candidates=[]
        if model=='iid_nb2':nested=[a,0.]; initial=[[a,v] for v in [.01,.05,.1,.3,1,3,7,9]]
        elif model=='iid_poisson_mix':nested=[a,a,0.];initial=[[a+u,a+v,t] for u,v,t in [(-.5,.3,0),(-1,.2,-1),(-.2,.7,1),(-.7,.7,0)]];initial += [[a+rng.normal(0,.5),a+rng.normal(0,.5),rng.normal()] for _ in range(6)]
        elif model=='poisson_ar':nested=[a,0.];initial=[[a-b*x.mean(),b] for b in [-.8,-.4,0,.2,.5,.8]]
        else:
            ar=fits['poisson_ar']['params'];im=fits['iid_poisson_mix']['params'];nested=[ar[0],ar[1],ar[0],ar[1],0.]
            initial=[nested,[im[0],0,im[1],0,im[2]]]
            initial += [[ar[0]+rng.normal(0,.4),np.clip(ar[1]+rng.normal(0,.2),-.9,.9),ar[0]+rng.normal(0,.4),np.clip(ar[1]+rng.normal(0,.2),-.9,.9),rng.normal()] for _ in range(8)]
        nested_obj=fits['poisson_ar']['objective'] if model=='poisson_ar_mix' else base
        nested_candidates=[(nested,nested_obj,'exact_nested')]
        if model=='poisson_ar_mix':
            im=fits['iid_poisson_mix']['params'];nested_candidates.append(([im[0],0,im[1],0,im[2]],fits['iid_poisson_mix']['objective'],'exact_iid_mixture'))
        for p,v,source in nested_candidates:
            rec={'source':source,'objective':v,'success':True,'iterations':0,'params':list(p),'boundary':['alpha=0'] if model=='iid_nb2' else []};starts.append(rec);candidates.append(rec)
        for j,p in enumerate(initial):
            p=np.clip(p,np.array(BOUNDS[model])[:,0],np.array(BOUNDS[model])[:,1])
            res=minimize(objective,p,args=(model,y,x),method='L-BFGS-B',jac=model!='iid_nb2',bounds=BOUNDS[model],options={'maxiter':1500,'ftol':1e-12,'gtol':1e-7,'maxls':40})
            boundary=[str(k) for k,(lo,hi) in enumerate(BOUNDS[model]) if min(abs(res.x[k]-lo),abs(res.x[k]-hi))<1e-5]
            rec={'source':f'start_{j}','initial':p.tolist(),'objective':float(res.fun),'success':bool(res.success),'status':int(res.status),'message':str(res.message),'iterations':int(res.nit),'params':res.x.tolist(),'boundary':boundary}
            starts.append(rec)
            if res.success and np.isfinite(res.fun):candidates.append(rec)
        best=min(candidates,key=lambda z:z['objective']);p=best['params'].copy()
        if model in ['iid_poisson_mix','poisson_ar_mix']:
            if model=='iid_poisson_mix':i,j,t=0,1,2;v1,v2=p[0],p[1]
            else:i,j,t=0,2,4;v1=p[0]+p[1]*np.log(13);v2=p[2]+p[3]*np.log(13)
            if v1>v2:
                if model=='iid_poisson_mix':p=[p[1],p[0],-p[2]]
                else:p=[p[2],p[3],p[0],p[1],-p[4]]
        fits[model]={'params':p,'objective':best['objective'],'selected_source':best['source'],'starts':starts,'failed_starts':sum(not z['success'] for z in starts),'selected_boundary':best['boundary']}
    return fits

def components(model,p,lags):
    if model=='iid_poisson':return [(1.,np.full(len(lags),np.exp(p[0])))]
    if model=='iid_nb2':return [(1.,np.full(len(lags),np.exp(p[0])))]
    if model=='poisson_ar':return [(1.,np.exp(p[0]+p[1]*np.log1p(lags)))]
    if model=='iid_poisson_mix':return [(expit(p[2]),np.full(len(lags),np.exp(p[0]))),(expit(-p[2]),np.full(len(lags),np.exp(p[1]))) ]
    return [(expit(p[4]),np.exp(p[0]+p[1]*np.log1p(lags))),(expit(-p[4]),np.exp(p[2]+p[3]*np.log1p(lags)))]

def kernel(model,p,lags,K):
    grid=np.arange(K+1);mat=np.zeros((len(lags),K+1));tail=np.zeros(len(lags));moment=tail.copy()
    for w,m in components(model,p,lags):
        if model=='iid_nb2' and p[1]>0:
            r=1/p[1];pr=r/(r+m)
            mat+=w*nbinom.pmf(grid[None,:],r,pr[:,None]);tail+=w*nbinom.sf(K,r,pr);moment+=w*m*nbinom.sf(K-1,r+1,pr)
        else:
            mat+=w*poisson.pmf(grid[None,:],m[:,None]);tail+=w*poisson.sf(K,m);moment+=w*m*poisson.sf(K-1,m)
    return mat,tail,moment

def metrics(p,y):
    k=np.arange(len(p));cdf=np.cumsum(p);mean=float(p@k)
    qs=[int(np.searchsorted(cdf,q)) for q in [.025,.1,.5,.9,.975]]
    prob=float(p[y]) if y<len(p) else 0.
    return {'probability':prob,'nll':float(-np.log(prob)) if prob>0 else None,'rps':float(np.sum((cdf-(k>=y))**2)),'mean':mean,'median':qs[2],'interval80':[qs[1],qs[3]],'interval95':[qs[0],qs[4]],'squared_error':(mean-y)**2,'absolute_error':abs(qs[2]-y),'coverage80':qs[1]<=y<=qs[3],'coverage95':qs[0]<=y<=qs[4],'width80':qs[3]-qs[1],'width95':qs[4]-qs[0]}

def predict_cap(model,p,lag,H,K):
    mat,tail,tm=kernel(model,p,np.arange(K+1),K)
    first,ft,fm=kernel(model,p,np.array([lag]),K)
    q=first[0];M=float(ft[0]);U=float(fm[0]);out=[]
    A=max(np.exp(p[0]),np.exp(p[2])) if model=='poisson_ar_mix' else (max(np.exp(p[0]),np.exp(p[1])) if model=='iid_poisson_mix' else np.exp(p[0]))
    for h in range(1,H+1):
        if h>1:
            escaped=float(q@tail);em=float(q@tm);U=A*(U+M)+em;M+=escaped;q=q@mat
        retained=float(q.sum());norm=q/retained;mean=float(norm@np.arange(K+1))
        out.append({'pmf':norm,'retained_pmf':q.copy(),'escaped_mass':M,'retained_mass':retained,'missing_first_moment_upper':U,'mean_error_bound':U+M*mean,'rps_error_bound':2*(K+1)*M+U,'cdf_error_bound':M,'roundoff_mass_discrepancy':abs(1-retained-M)})
    return out

def forecast(model,p,lag,H):
    """Select numerical cap using fitted parameters and final training lag only."""
    K=64
    while K<=2048:
        small=predict_cap(model,p,lag,H,K);big=predict_cap(model,p,lag,H,2*K);ok=True
        for h,(s,b) in enumerate(zip(small,big)):
            ms=metrics(s['pmf'],0);mb=metrics(b['pmf'],0)
            diff={'mean':abs(ms['mean']-mb['mean']),'rps':abs(ms['rps']-mb['rps']),'quantiles_equal':all(ms[k]==mb[k] for k in ['median','interval80','interval95'])}
            diff['rps_reference_count']=0
            b['doubling_check']=diff;b['cap']=2*K
            if b['escaped_mass']>1e-12 or b['mean_error_bound']>1e-8 or b['rps_error_bound']>1e-8 or diff['mean']>1e-8 or diff['rps']>1e-8 or not diff['quantiles_equal']:ok=False
        if ok:
            for b in big:b['past_only_cap_checks_passed']=True
            return big
        K*=2
    raise RuntimeError(f'Numerical tolerances unmet at cap4096: {model}')

def summarize(rows):
    out={}
    for model in MODELS:
        rr=[r for r in rows if r['model']==model]
        out[model]={k:float(np.mean([r[k] for r in rr])) for k in ['nll','rps','absolute_error','coverage80','coverage95','width80','width95']}
        out[model]['rmse']=float(np.sqrt(np.mean([r['squared_error'] for r in rr])))
        out[model]['cases']=len(rr)
    return out

def checks(data):
    one=forecast('poisson_ar',[1.,.3],12,1)[0]['pmf'];exact=poisson.pmf(np.arange(len(one)),np.exp(1+.3*np.log(13)));assert np.max(abs(one-exact))<1e-12
    ii=forecast('iid_poisson',[np.log(12)],12,8);assert max(np.max(abs(z['pmf']-ii[0]['pmf'])) for z in ii)<1e-12
    mix=forecast('poisson_ar_mix',[1.,.3,1.,.3,.7],12,2);ar=forecast('poisson_ar',[1.,.3],12,2);assert max(np.max(abs(z['pmf']-v['pmf'])) for z,v in zip(mix,ar))<1e-12
    nb=kernel('iid_nb2',[np.log(12),1e-8],np.array([12]),128)[0][0];assert np.max(abs(nb-poisson.pmf(np.arange(129),12)))<1e-6
    pp=np.zeros(20);pp[4]=1;assert metrics(pp,10)['rps']==6
    for origin in range(1977,2013):
        training=data[:origin-1918+1];assert len(training)>=60 and 1918+len(training)-1==origin and origin+8<=2020
    import inspect
    assert list(inspect.signature(forecast).parameters)==['model','p','lag','H']
    return {'one_step':'passed','iid_horizons':'passed','collapsed_mixture':'passed','nb_poisson_limit':'passed','rps_point_mass':'passed','calendar_alignment':'passed','training_slice_no_future':'passed','forecast_api_has_no_outcome_argument':'passed'}

def main():
    start=time.time();data=read_counts();tests=checks(data);rows=[];fits={}
    for origin in range(1977,2013):
        train=data[:origin-1918+1];ff=fit_all(train,origin);fits[str(origin)]=ff
        for model in MODELS:
            pred=forecast(model,ff[model]['params'],int(train[-1]),8)
            targets=data[origin-1918+1:origin-1918+9]
            for h,(pr,target) in enumerate(zip(pred,targets),1):
                if not 0<=target<len(pr['pmf']):raise RuntimeError('Observed count outside past-only forecast cap')
                mm=metrics(pr['pmf'],int(target));q=float(pr['retained_pmf'][target]);M=pr['escaped_mass']
                if not (np.isfinite(q) and q>0 and np.isfinite(mm['probability']) and mm['probability']>0 and np.isfinite(mm['nll'])):raise RuntimeError('Invalid observed predictive probability')
                bracket=[float(-np.log(min(1,q+M))),float(-np.log(q))]
                floating_tolerance=1e-8
                log_error_bound=float(abs(np.log(pr['retained_mass']))+np.log1p(M/q))
                if log_error_bound>floating_tolerance or not bracket[0]-floating_tolerance<=mm['nll']<=bracket[1]+floating_tolerance:raise RuntimeError('Observed log-score numerical guard failed')
                diagnostic={k:v for k,v in pr.items() if k not in ['pmf','retained_pmf']}
                diagnostic.update({'true_nll_bracket':bracket,'observed_probability_guard_passed':True,'observed_nll_bracket_guard_passed':True,'observed_nll_error_bound_with_mass_roundoff':log_error_bound,'nll_guard_floating_tolerance':floating_tolerance})
                rows.append({'origin':origin,'horizon':h,'target_year':origin+h,'target':int(target),'model':model,**mm,'numerics':diagnostic})
        print(f'origin {origin} complete ({time.time()-start:.1f}s)',flush=True)
    ff=fit_all(data,2020);frozen={'schema':1,'origin':2020,'seed':SEED,'code_sha256':sha(__file__),'data_sha256':sha(ROOT/'usgs_1918_2020_snapshot.json'),'fits':ff,'forecasts':{}}
    for model in MODELS:
        pr=forecast(model,ff[model]['params'],int(data[-1]),8)
        frozen['forecasts'][model]=[{'horizon':h,'target_year':2020+h,**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items() if k!='retained_pmf'}} for h,p in enumerate(pr,1)]
    payload=json.dumps(frozen,indent=2,allow_nan=False)+'\n';fp=ROOT/'forecast_2020_frozen.json'
    if fp.exists():
        if fp.read_text(encoding='utf-8-sig')!=payload:raise RuntimeError('Existing frozen file differs; refusing overwrite')
    else:fp.write_text(payload,encoding='utf-8')
    settings={'origins':[1977,2012],'horizons':[1,8],'training_response':'second through final training year for every model','seed':SEED,'bounds':BOUNDS,'component_label_reference_lag':12,'optimizer':'L-BFGS-B training conditional likelihood; deterministic multiple starts; exact nested candidates','failure_policy':'Exclude failed optimizer starts, retain explicit successful nested candidate; never omit forecast cases. Finite local search gives no global optimum guarantee.','prediction':'Normalized retained path PMF; adaptive cap uses past data only, analytic bounds, mean, RPS at fixed reference count zero, and quantile cap-doubling checks. Bounds apply in exact arithmetic; reported roundoff mass discrepancy measures floating point normalization drift, not a certified rounding bound.','tolerances':{'escaped_mass':1e-12,'mean_error_bound':1e-8,'rps_error_bound':1e-8,'doubling_mean_rps':1e-8,'observed_nll_guard_floating_tolerance':1e-8},'parameter_uncertainty':'not included','intervals':'inclusive, discrete equal-tailed 80% and 95%','primary_score':'negative log predictive probability','rps':'sum over nonnegative integers of squared CDF minus observation-step difference, normalized PMF','raw_intercept_bound':[-8,8],'weight_logit_bound':[-12,12],'nb_alpha_bound':[0,10],'alpha_zero':'exact Poisson candidate; positive optimizer alpha lower bound 1e-8'}
    paired=[]
    for origin in range(1977,2013):
        rr=[r for r in rows if r['origin']==origin];su=summarize(rr);base=su['iid_poisson']
        paired.append({'origin':origin,'differences_vs_iid_poisson':{m:{s:su[m][s]-base[s] for s in ['nll','rps']} for m in MODELS}})
    result={'settings':settings,'checks':tests,'annual_counts':{str(1918+i):int(v) for i,v in enumerate(data)},'data_sha256':sha(ROOT/'usgs_1918_2020_snapshot.json'),'code_sha256':sha(__file__),'frozen_sha256':sha(fp),'summary':summarize(rows),'by_horizon':{str(h):summarize([r for r in rows if r['horizon']==h]) for h in range(1,9)},'fixed_origin_split':{'first18':summarize([r for r in rows if r['origin']<=1994]),'last18':summarize([r for r in rows if r['origin']>=1995])},'paired_origin_score_differences':paired,'fits':fits,'rows':rows,'elapsed_seconds':time.time()-start}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result['summary'],indent=2));print('elapsed_seconds',result['elapsed_seconds']);print('code_sha256',result['code_sha256']);print('frozen_sha256',result['frozen_sha256'])
if __name__=='__main__':main()



import numpy as np
import pandas as pd
import pytest
from src.retention import check_history, fit_retention, score_active, select_features

def data(n=140):
    rng=np.random.default_rng(11)
    tenure=rng.integers(1,72,n)
    contract=rng.choice(['Monthly','Annual'],n)
    p=1/(1+np.exp(-(-.3 + .9*(contract=='Monthly') - .025*tenure)))
    return pd.DataFrame({'CustomerID':[f'X{i}' for i in range(n)],'TenureMonths':tenure,'ContractType':contract,'MonthlyRevenue':rng.uniform(20,100,n),'Churn':np.where(rng.random(n)<p,'Yes','No'),'Leakage_Do_Not_Use':np.where(p>.5,'closed','open')})

def test_leakage_excluded():
    assert 'Leakage_Do_Not_Use' not in select_features(data())

def test_training():
    r=fit_retention(data())
    assert 0<=r['metrics']['ROC AUC']<=1
    assert 'Leakage_Do_Not_Use' not in r['features']

def test_scoring():
    d=data();r=fit_retention(d)
    active=d.drop(columns=['Churn']).iloc[:12].copy()
    out=score_active(r,active)
    assert len(out)==12
    assert out.ExpectedMonthlyRevenueAtRisk.ge(0).all()
    assert out.PriorityScore.is_monotonic_decreasing

def test_missing_schema():
    with pytest.raises(ValueError):check_history(data().drop(columns=['CustomerID']))

def test_active_requires_features():
    r=fit_retention(data())
    with pytest.raises(ValueError):score_active(r,data()[['CustomerID','MonthlyRevenue']])

def test_duplicate_ids_rejected():
    r=fit_retention(data());d=data().iloc[:5].copy();d.loc[d.index[1],'CustomerID']=d.iloc[0].CustomerID
    with pytest.raises(ValueError):score_active(r,d)

"""Customer retention modeling with explicit temporal and leakage safeguards."""
import io
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance

TARGET = 'Churn'
IDENTITY = 'CustomerID'
REVENUE = 'MonthlyRevenue'
BLOCKED_WORDS = ('churn', 'cancel', 'termination', 'terminated', 'closed', 'exit', 'outcome', 'target', 'leakage', 'postevent', 'post_event')

def read_csv(contents):
    df = pd.read_csv(io.BytesIO(contents))
    df.columns = df.columns.astype(str).str.strip()
    if df.empty or not df.columns.is_unique:
        raise ValueError('CSV must contain data and unique column names.')
    return df

def check_history(df):
    required = {IDENTITY, REVENUE, TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError('Historical CSV missing: ' + ', '.join(sorted(missing)))
    d = df.copy()
    d[TARGET] = d[TARGET].astype('string').str.strip().str.lower().map({'yes':1,'no':0,'1':1,'0':0,'true':1,'false':0,'exited':1,'stayed':0})
    d = d.dropna(subset=[TARGET, IDENTITY]).drop_duplicates(subset=[IDENTITY], keep='first')
    d[REVENUE] = pd.to_numeric(d[REVENUE], errors='coerce')
    if len(d)<60 or d[TARGET].nunique()!=2 or d[TARGET].value_counts().min()<12:
        raise ValueError('Need at least 60 labeled unique customers and 12 examples of each outcome.')
    return d

def select_features(df):
    """Conservative allowlist: never learn arbitrary columns or post-outcome fields."""
    allowed = ['TenureMonths','ContractType','SupportTickets','PaymentMethod','InternetService','MonthlyRevenue']
    features = [c for c in allowed if c in df.columns]
    if len(features)<2:
        raise ValueError('Need at least two supported features: ' + ', '.join(allowed))
    return features

def pipeline_for(X, estimator):
    num = X.select_dtypes(include='number').columns.tolist()
    cat = [c for c in X.columns if c not in num]
    parts=[]
    if num:
        parts.append(('numeric', Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num))
    if cat:
        parts.append(('category', Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat))
    return Pipeline([('prepare',ColumnTransformer(parts)),('model',estimator)])

def fit_retention(history):
    d = check_history(history)
    features=select_features(d)
    X=d[features].copy(); y=d[TARGET].astype(int)
    for c in X.select_dtypes(exclude='number'):
        X[c]=X[c].map(lambda v: str(v).strip().lower() if pd.notna(v) and str(v).strip() else np.nan).astype(object)
    xtr,xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    candidates={'Logistic regression':LogisticRegression(max_iter=2000), 'Random forest':RandomForestClassifier(n_estimators=120,min_samples_leaf=3,random_state=42,n_jobs=1)}
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    scores={name:float(cross_val_score(pipeline_for(xtr,est),xtr,ytr,cv=cv,scoring='roc_auc').mean()) for name,est in candidates.items()}
    winner=max(scores,key=scores.get)
    model=pipeline_for(xtr,candidates[winner]);model.fit(xtr,ytr)
    p=model.predict_proba(xte)[:,1];pred=(p>=.5).astype(int)
    metrics={'ROC AUC':float(roc_auc_score(yte,p)), 'Recall':float(recall_score(yte,pred,zero_division=0)), 'Precision':float(precision_score(yte,pred,zero_division=0)), 'F1':float(f1_score(yte,pred,zero_division=0)), 'Accuracy':float(accuracy_score(yte,pred))}
    imp=permutation_importance(model,xte,yte,n_repeats=3,random_state=42,scoring='roc_auc')
    importance=pd.DataFrame({'Feature':features,'Importance':imp.importances_mean}).sort_values('Importance',ascending=False)
    return {'model':model,'features':features,'scores':scores,'winner':winner,'metrics':metrics,'matrix':confusion_matrix(yte,pred,labels=[0,1]),'importance':importance,'test_rows':len(xte),'train_rows':len(xtr),'history_rows':len(d),'churn_rate':float(y.mean())}

def score_active(result, active):
    if IDENTITY not in active or REVENUE not in active:
        raise ValueError('Active CSV needs CustomerID and MonthlyRevenue.')
    missing=set(result['features'])-set(active.columns)
    if missing:
        raise ValueError('Active CSV missing model inputs: '+', '.join(sorted(missing)))
    if active[IDENTITY].isna().any() or active[IDENTITY].duplicated().any():
        raise ValueError('Active CustomerID values must be present and unique.')
    X=active[result['features']].copy()
    for c in X.select_dtypes(exclude='number'):
        X[c]=X[c].map(lambda v: str(v).strip().lower() if pd.notna(v) and str(v).strip() else np.nan).astype(object)
    p=result['model'].predict_proba(X)[:,1]
    revenue=pd.to_numeric(active[REVENUE],errors='coerce')
    if revenue.isna().any() or (revenue<0).any():
        raise ValueError('MonthlyRevenue must contain nonnegative numbers for every active customer.')
    output=pd.DataFrame({'CustomerID':active[IDENTITY].to_numpy(),'ChurnProbability':p,'MonthlyRevenue':revenue.to_numpy()})
    output['ExpectedMonthlyRevenueAtRisk']=output.ChurnProbability*output.MonthlyRevenue
    output['RiskBand']=pd.cut(output.ChurnProbability,bins=[-0.01,.35,.65,1],labels=['Low','Medium','High'])
    output['PriorityScore']=output.ExpectedMonthlyRevenueAtRisk
    return output.sort_values('PriorityScore',ascending=False).reset_index(drop=True)

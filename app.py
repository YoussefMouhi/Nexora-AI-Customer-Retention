"""Nexora AI — Customer Retention Intelligence."""
import hashlib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.retention import read_csv, fit_retention, score_active

st.set_page_config(page_title='Nexora AI | Retention Intelligence',page_icon="assets/nexora_icon.png",layout='wide')

# Nexora AI branding
col_logo, col_title = st.columns(
    [1, 10],
    vertical_alignment="center"
)

with col_logo:
    st.image("assets/nexora_icon.png", width=55)

with col_title:
    st.title("Nexora AI — Customer Retention Intelligence")



st.markdown('''<style>
.stApp{background:#f5f8fc;color:#173352}section[data-testid="stSidebar"]{background:#122e4e}section[data-testid="stSidebar"] *{color:#f0f6ff!important}h1,h2,h3{color:#173352!important}.block-container{max-width:1250px;padding-top:5rem}.hero{background:linear-gradient(110deg,#123b69,#1487a8);padding:28px 32px;border-radius:16px;color:white;margin-bottom:20px}.hero h1{color:white!important;margin:0}.hero p{color:#e2f3ff;margin:8px 0 0}</style>''',unsafe_allow_html=True)
# Sidebar upload widgets: Streamlit's light drop zone needs explicit dark-on-light text.
st.markdown("""<style>
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background: #eaf2fb !important; border: 2px dashed #7ba5d4 !important; border-radius: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileSize"] {
  color: #163653 !important; opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
  background: #ffffff !important; color: #163653 !important; border: 1px solid #93aac3 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {color:#163653 !important;}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
  background: #ffffff !important; border-radius: 8px !important; padding: 4px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] button {
  background: #edf3fa !important; border-radius: 6px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] button svg {fill:#163653 !important; stroke:#163653 !important;}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg {color:#163653 !important;}
</style>""", unsafe_allow_html=True)
st.caption('Decision-support prototype • Probabilities are estimates, not guaranteed cancellations or recovered revenue.')

with st.sidebar:
    st.header('1 · Historical customers')
    history_file=st.file_uploader('Labeled training CSV',type=['csv'],key='history',help='CustomerID, Churn, MonthlyRevenue and at least two supported predictors.')
    st.header('2 · Active customers')
    active_file=st.file_uploader('Customers to prioritize',type=['csv'],key='active',help='Current customers without churn labels. Must contain the model input columns.')
    st.caption('Start with the sample files in examples/ if you want to test the workflow.')
    train_button=st.button('Train retention model',type='primary',use_container_width=True,disabled=history_file is None)

if history_file is None:
    st.info('Upload historical customer data to begin. No default dataset is loaded.')
    st.markdown('### How it works')
    st.markdown('**1.** Train on past customer outcomes.  **2.** Evaluate on held-out customers.  **3.** Score active customers.  **4.** Export a prioritized outreach list.')
    st.stop()

try:
    history_bytes=history_file.getvalue()
    history=read_csv(history_bytes)
except Exception as exc:
    st.error(f'Cannot read historical CSV: {exc}');st.stop()

fingerprint=hashlib.sha256(history_bytes).hexdigest()
if st.session_state.get('trained_on')!=fingerprint:
    st.session_state.pop('retention_result',None)

st.markdown('### Historical data overview')
a,b,c=st.columns(3)
a.metric('Uploaded records',f'{len(history):,}')
b.metric('Columns',len(history.columns))
c.metric('Missing cells',f'{int(history.isna().sum().sum()):,}')
with st.expander('Inspect data and supported feature schema'):
    st.dataframe(history.head(15),use_container_width=True)
    st.caption('Supported predictors: TenureMonths, ContractType, SupportTickets, PaymentMethod, InternetService, MonthlyRevenue. Other columns are ignored, including post-churn fields. CustomerID is used only for identification.')

if train_button:
    try:
        with st.spinner('Comparing models with 3-fold cross-validation...'):
            result=fit_retention(history)
        st.session_state['retention_result']=result
        st.session_state['trained_on']=fingerprint
    except Exception as exc:
        st.error(f'Training could not complete: {exc}')

result=st.session_state.get('retention_result')
if not result:
    st.info('Click **Train retention model** in the sidebar to evaluate historical outcomes.')
    st.stop()

st.success(f"Model trained: {result['winner']} · {result['train_rows']} training records · {result['test_rows']} held-out test records")
tab1,tab2,tab3=st.tabs(['Business overview','Customer priorities','Model reliability'])
with tab1:
    st.subheader('Retention overview')
    if active_file is None:
        st.info('Upload a separate active-customer CSV to estimate churn risk and prioritize outreach. Historical customers are not treated as active leads.')
    else:
        try:
            active=read_csv(active_file.getvalue())
            scored=score_active(result,active)
            st.session_state['scored']=scored
            st.session_state['active_hash']=hashlib.sha256(active_file.getvalue()).hexdigest()
            a,b,c,d=st.columns(4)
            a.metric('Active customers',f'{len(scored):,}')
            b.metric('High-risk customers',f"{int((scored.RiskBand=='High').sum()):,}")
            c.metric('Expected monthly revenue at risk',f"${scored.ExpectedMonthlyRevenueAtRisk.sum():,.0f}")
            d.metric('Average churn probability',f"{scored.ChurnProbability.mean():.1%}")
            st.plotly_chart(px.histogram(scored,x='ChurnProbability',nbins=20,title='Predicted churn risk distribution',labels={'ChurnProbability':'Churn probability'}),use_container_width=True)
            st.caption('Expected revenue at risk = sum of predicted churn probability × monthly revenue. This is a planning estimate, not a forecast of realized losses or intervention impact.')
        except Exception as exc:
            st.session_state.pop('scored',None)
            st.error(f'Unable to score active customers: {exc}')
with tab2:
    scored=st.session_state.get('scored')
    if scored is None or active_file is None or st.session_state.get('active_hash')!=hashlib.sha256(active_file.getvalue()).hexdigest():
        st.info('Upload valid active-customer data to generate an outreach list.')
    else:
        st.subheader('Prioritized outreach list')
        band=st.selectbox('Filter by risk', ['All','High','Medium','Low'])
        view=scored if band=='All' else scored[scored.RiskBand.astype(str)==band]
        st.dataframe(view.style.format({'ChurnProbability':'{:.1%}','MonthlyRevenue':'${:,.2f}','ExpectedMonthlyRevenueAtRisk':'${:,.2f}','PriorityScore':'{:,.2f}'}),use_container_width=True,hide_index=True)
        st.download_button('Download outreach priorities (CSV)',data=view.to_csv(index=False).encode(),file_name='nexora_retention_priorities.csv',mime='text/csv')
        st.caption('Priority ranks expected revenue exposure, not proven response to a retention campaign. Test interventions before claiming saved revenue.')
with tab3:
    st.subheader('Model reliability')
    st.caption('Candidate models are selected using training-only cross-validation. Metrics below are from a separate held-out test set.')
    cols=st.columns(5)
    for col,(name,value) in zip(cols,result['metrics'].items()):
        col.metric(name,f'{value:.3f}')
    st.dataframe(pd.DataFrame({'Model':list(result['scores']),'CV ROC AUC':list(result['scores'].values())}),hide_index=True,use_container_width=True)
    left,right=st.columns(2)
    with left:
        cm=result['matrix']
        st.plotly_chart(px.imshow(cm,text_auto=True,x=['Predicted stayed','Predicted churn'],y=['Actual stayed','Actual churn'],title='Confusion matrix',color_continuous_scale='Blues'),use_container_width=True)
    with right:
        st.plotly_chart(px.bar(result['importance'],x='Importance',y='Feature',orientation='h',title='Permutation importance (held-out ROC AUC)').update_layout(yaxis={'categoryorder':'total ascending'}),use_container_width=True)
    st.markdown('**Features used:** '+', '.join(result['features']))
    st.warning('Safeguards: only documented pre-outcome features are allowed. This reduces accidental leakage but cannot verify when each value was collected. Historical data must represent the same population and time horizon as active customers.')
    st.caption('Thresholds: Low <35%, Medium 35–65%, High ≥65%. These are illustrative operating bands, not optimized decision thresholds. Review class balance, calibration, fairness and business costs before operational use.')

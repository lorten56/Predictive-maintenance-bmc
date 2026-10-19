from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title='Predictive Maintenance Digital Twin', page_icon='⚙️', layout='wide')
ROOT = Path(__file__).resolve().parent / 'data'
st.title('⚙️ Predictive Maintenance Digital Twin')
st.caption('Synthetic 2021–2025 historical demonstration • Not a live or validated operational monitoring system')

@st.cache_data
def load_csv(name):
    df = pd.read_csv(ROOT / name, parse_dates=['Date'])
    return df

@st.cache_resource
def train_model():
    df = load_csv('prediction_dataset.csv').sort_values('Date').reset_index(drop=True)
    target = 'Failure_Within_Next_7_Days'
    features = [c for c in df.columns if c not in ['Date', target]]
    numeric = [c for c in features if c != 'Machine']
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), numeric),
        ('cat', Pipeline([('fill', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), ['Machine'])
    ])
    model = Pipeline([('preprocess', pre), ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=8, class_weight='balanced_subsample', random_state=42, n_jobs=-1))])
    train = df[df.Date < '2024-01-01']
    valid = df[(df.Date >= '2024-01-01') & (df.Date < '2025-01-01')]
    test = df[df.Date >= '2025-01-01'].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError('Expected 2021–2023 train, 2024 validation, and 2025 test records.')
    model.fit(train[features], train[target])
    val_prob = model.predict_proba(valid[features])[:, 1]
    thresholds = np.arange(.05, .81, .05)
    chosen = max(thresholds, key=lambda t: f1_score(valid[target], val_prob >= t, zero_division=0))
    test['Predicted_Risk'] = model.predict_proba(test[features])[:, 1]
    return df, test, model, features, float(chosen)

try:
    prediction, test, model, features, validation_threshold = train_model()
    components = load_csv('component_condition.csv')
    failures = load_csv('failure_log.csv')
    maintenance = load_csv('maintenance_history.csv')
    production = load_csv('production_and_downtime.csv')
except Exception as e:
    st.error(f'Unable to load dashboard data: {e}')
    st.stop()

machines = sorted(prediction.Machine.dropna().unique())
with st.sidebar:
    st.header('Dashboard filters')
    selected = st.selectbox('Machine', ['All machines'] + machines)
    threshold = st.slider('Risk alert threshold', .05, .95, float(round(validation_threshold, 2)), .05)
    st.caption(f'Validation-selected threshold: {validation_threshold:.2f}. Changing this slider is exploratory.')
    st.caption('Data are synthetic; dates reflect historical records, not current monitoring.')

def filtered(df):
    return df if selected == 'All machines' else df[df.Machine == selected]

p = filtered(production)
f = filtered(failures)
m = filtered(maintenance)
c = filtered(components)
t = filtered(test)
latest = t.sort_values('Date').groupby('Machine').tail(1).copy()
latest['Risk (%)'] = latest.Predicted_Risk * 100
latest['Alert'] = latest.Predicted_Risk >= threshold

pages = st.tabs(['Overview', 'Failure prediction', 'Component condition', 'Reliability & downtime', 'Maintenance history'])
with pages[0]:
    a,b,d,e = st.columns(4)
    a.metric('Machines in view', p.Machine.nunique())
    b.metric('Historical failures', len(f))
    d.metric('Recorded downtime (hours)', f'{p.Downtime_Hours.sum():,.1f}')
    e.metric('Latest test-period risk alerts', int(latest.Alert.sum()))
    st.subheader('Historical output efficiency')
    efficiency = p.groupby('Date', as_index=False).Output_Efficiency.mean()
    st.plotly_chart(px.line(efficiency, x='Date', y='Output_Efficiency'), use_container_width=True)
    st.subheader('Latest 2025 test-period risk per machine')
    st.plotly_chart(px.bar(latest, x='Machine', y='Risk (%)', color='Alert', color_discrete_map={True:'#dc3545', False:'#16825d'}), use_container_width=True)

with pages[1]:
    st.subheader('Random Forest: seven-day failure risk')
    st.info('Model trained on 2021–2023, threshold selected on 2024 validation, evaluated on 2025 holdout. The data are synthetic. Historical test predictions are not live forecasts.')
    evaluated = (t.Predicted_Risk >= threshold).astype(int)
    truth = t.Failure_Within_Next_7_Days.astype(int)
    a,b,d,e = st.columns(4)
    a.metric('Precision', f'{precision_score(truth,evaluated,zero_division=0):.3f}')
    b.metric('Recall', f'{recall_score(truth,evaluated,zero_division=0):.3f}')
    d.metric('F1', f'{f1_score(truth,evaluated,zero_division=0):.3f}')
    e.metric('PR-AUC', f'{average_precision_score(truth,t.Predicted_Risk):.3f}' if truth.nunique()>1 else 'N/A')
    cm = confusion_matrix(truth, evaluated, labels=[0,1])
    st.plotly_chart(px.imshow(cm, text_auto=True, x=['Predicted no failure','Predicted failure'], y=['Actual no failure','Actual failure'], title='2025 test confusion matrix'), use_container_width=True)
    st.plotly_chart(px.line(t.sort_values('Date'), x='Date', y='Predicted_Risk', color='Machine', title='Historical test-period risk'), use_container_width=True)
    st.dataframe(latest[['Date','Machine','Risk (%)','Alert']].sort_values('Risk (%)',ascending=False), use_container_width=True, hide_index=True)

with pages[2]:
    st.subheader('Component health and wear')
    latest_components = c.sort_values('Date').groupby(['Machine','Component']).tail(1)
    st.plotly_chart(px.bar(latest_components, x='Component', y='Wear_Percentage', color='Machine', title='Latest historical wear (%)'), use_container_width=True)
    st.plotly_chart(px.line(c.groupby('Date',as_index=False).Estimated_Health_Score.mean(), x='Date', y='Estimated_Health_Score', title='Average component health over time'), use_container_width=True)
    st.dataframe(latest_components[['Date','Machine','Component','Wear_Percentage','Estimated_Health_Score','Days_Since_Last_Maintenance']].sort_values('Wear_Percentage',ascending=False), use_container_width=True, hide_index=True)

with pages[3]:
    st.subheader('Production and downtime')
    summary = p.groupby('Machine',as_index=False).agg(Downtime_Hours=('Downtime_Hours','sum'), Operating_Hours=('Operating_Hours','sum'), Failures=('Failure_Count','sum'), Mean_Output_Efficiency=('Output_Efficiency','mean'))
    summary['Observed_Availability_Proxy'] = summary.Operating_Hours / (summary.Operating_Hours + summary.Downtime_Hours).replace(0,np.nan)
    st.caption('Availability proxy uses reported operating and downtime hours. This is not independently verified OEE availability; MTBF/MTTR are not calculated without validated event/exposure definitions.')
    st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(summary, x='Machine', y='Downtime_Hours', title='Cumulative downtime (hours)'), use_container_width=True)
    st.plotly_chart(px.line(p.groupby('Date',as_index=False).Downtime_Hours.sum(), x='Date', y='Downtime_Hours', title='Daily downtime'), use_container_width=True)

with pages[4]:
    st.subheader('Failures and maintenance')
    a,b = st.columns(2)
    a.metric('Failure events',len(f))
    b.metric('Maintenance actions',len(m))
    if not f.empty:
        counts = f.Failure_Mode.value_counts().rename_axis('Failure_Mode').reset_index(name='Count')
        st.plotly_chart(px.bar(counts, x='Failure_Mode', y='Count', title='Failure mode frequency'), use_container_width=True)
    if not m.empty:
        st.plotly_chart(px.pie(m, names='Maintenance_Type', title='Maintenance type distribution'), use_container_width=True)
        st.dataframe(m.sort_values('Date',ascending=False).head(100), use_container_width=True, hide_index=True)

st.divider()
st.caption('Prototype decision support only. Synthetic records; review data leakage, model calibration, and independent validation before operational use.')

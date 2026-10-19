from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title='Predictive Maintenance Digital Twin', page_icon='⚙️', layout='wide')
ROOT = Path(__file__).resolve().parent / 'data'
st.title('⚙️ Predictive Maintenance Digital Twin')
st.caption('Historical synthetic demonstration (2021–2025). Predictions, survival estimates and costs are not validated for operational use.')

@st.cache_data
def load_csv(filename):
    return pd.read_csv(ROOT / filename, parse_dates=['Date'])

@st.cache_resource(show_spinner='Comparing predictive models on historical validation data...')
def model_bundle():
    df = load_csv('prediction_dataset.csv').sort_values('Date').reset_index(drop=True)
    target = 'Failure_Within_Next_7_Days'
    # Purge a full target horizon between date partitions to avoid overlapping seven-day labels.
    train = df[df.Date < '2023-12-25'].copy()
    val = df[(df.Date >= '2024-01-01') & (df.Date < '2024-12-25')].copy()
    test = df[df.Date >= '2025-01-01'].copy()
    features = [c for c in df.columns if c not in ('Date', target)]
    numeric = [c for c in features if c != 'Machine']
    pre = ColumnTransformer([
        ('numeric', Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('machine', OneHotEncoder(handle_unknown='ignore'), ['Machine'])
    ])
    candidates = {
        'Random Forest (balanced)': RandomForestClassifier(n_estimators=220, min_samples_leaf=4, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        'Logistic Regression (balanced)': LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42),
        'Gradient Boosting': HistGradientBoostingClassifier(max_iter=160, learning_rate=0.055, max_leaf_nodes=15, l2_regularization=2, random_state=42)
    }
    rows, fitted = [], {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([('preprocess', pre), ('classifier', classifier)])
        pipeline.fit(train[features], train[target])
        vp = pipeline.predict_proba(val[features])[:, 1]
        thresholds = np.arange(0.05, 0.96, 0.05)
        best_threshold = max(thresholds, key=lambda x: f1_score(val[target], vp >= x, zero_division=0))
        rows.append({'Model': name, 'Validation PR-AUC': average_precision_score(val[target], vp), 'Validation F1': f1_score(val[target], vp >= best_threshold, zero_division=0), 'Threshold': float(best_threshold)})
        fitted[name] = pipeline
    scores = pd.DataFrame(rows).sort_values(['Validation PR-AUC','Validation F1'], ascending=False)
    winner = scores.iloc[0]
    model = fitted[winner['Model']]
    test['Predicted_Risk'] = model.predict_proba(test[features])[:, 1]
    return df, test, scores, str(winner['Model']), float(winner['Threshold'])

try:
    prediction, test, model_scores, model_name, chosen_threshold = model_bundle()
    components = load_csv('component_condition.csv')
    failures = load_csv('failure_log.csv')
    maintenance = load_csv('maintenance_history.csv')
    production = load_csv('production_and_downtime.csv')
except Exception as error:
    st.error(f'Unable to initialize dashboard: {error}')
    st.stop()

machines = sorted(prediction.Machine.dropna().unique().tolist())
with st.sidebar:
    st.header('Filters and assumptions')
    selected = st.selectbox('Machine', ['All machines'] + machines)
    threshold = st.slider('Failure alert threshold', 0.05, 0.95, round(chosen_threshold, 2), 0.05)
    st.caption(f'Chosen on 2024 validation: {chosen_threshold:.2f}. Changing it is exploratory, not independent evaluation.')
    st.caption('Records are synthetic and historical, not live sensor readings.')

def filtered(df):
    return df if selected == 'All machines' else df.loc[df.Machine == selected].copy()

p, f, m, c, t = (filtered(d) for d in (production, failures, maintenance, components, test))
latest = t.sort_values('Date').groupby('Machine').tail(1).copy()
latest['Risk (%)'] = latest.Predicted_Risk * 100
latest['Alert'] = latest.Predicted_Risk >= threshold

# Build right-censored observed time-to-first-failure per component (starting at first observation).
# This is an observational first-event Kaplan–Meier approximation; resets/replacements are not inferred.
@st.cache_data
def survival_cohorts(condition, failure_log):
    rows = []
    failures_by_component = failure_log.groupby(['Machine','Component'])
    for (machine, component), g in condition.groupby(['Machine','Component']):
        g = g.sort_values('Date')
        start, end = g.Date.iloc[0], g.Date.iloc[-1]
        key = (machine, component)
        if key in failures_by_component.groups:
            dates = failure_log.loc[failures_by_component.groups[key], 'Date']
            observed = dates[(dates >= start) & (dates <= end)]
        else:
            observed = pd.Series(dtype='datetime64[ns]')
        event = not observed.empty
        stop = observed.min() if event else end
        rows.append({'Machine':machine, 'Component':component, 'Duration_Days':max(1, (stop-start).days+1), 'Event':int(event)})
    return pd.DataFrame(rows)

cohorts = survival_cohorts(components, failures)

def km_curve(records):
    if records.empty:
        return pd.DataFrame({'Day':[0], 'Survival':[1.0], 'At_Risk':[0]})
    durations = records.Duration_Days.to_numpy()
    events = records.Event.to_numpy()
    points = [{'Day':0, 'Survival':1.0, 'At_Risk':len(records)}]
    survival = 1.0
    for day in sorted(set(durations[events == 1])):
        at_risk = int((durations >= day).sum())
        n_events = int(((durations == day) & (events == 1)).sum())
        survival *= 1 - n_events / at_risk
        points.append({'Day':int(day), 'Survival':survival, 'At_Risk':at_risk})
    return pd.DataFrame(points)

pages = st.tabs(['Overview', 'Failure prediction', 'Component status', 'Reliability (MTBF / MTTR)', 'Survival & replacement cost', 'Maintenance history'])
with pages[0]:
    a,b,d,e = st.columns(4)
    a.metric('Machines in view', p.Machine.nunique())
    b.metric('Recorded failures', len(f))
    d.metric('Downtime (hours)', f'{p.Downtime_Hours.sum():,.1f}')
    e.metric('Latest test risk alerts', int(latest.Alert.sum()))
    st.plotly_chart(px.bar(latest, x='Machine', y='Risk (%)', color='Alert', title='Latest historical test risk by machine'), use_container_width=True)
    st.plotly_chart(px.line(p.groupby('Date',as_index=False).Output_Efficiency.mean(), x='Date', y='Output_Efficiency', title='Average historical output efficiency'), use_container_width=True)

with pages[1]:
    st.subheader('Model comparison and seven-day failure risk')
    st.dataframe(model_scores.round(3), hide_index=True, use_container_width=True)
    st.caption(f'Best validation PR-AUC: **{model_name}**. Training: through Dec 24, 2023; validation: Jan–Dec 24, 2024; test: 2025. Seven-day gap before validation and test. No model can guarantee higher accuracy on new data.')
    truth = t.Failure_Within_Next_7_Days.astype(int)
    predicted = (t.Predicted_Risk >= threshold).astype(int)
    cols = st.columns(5)
    metrics = [('Accuracy', accuracy_score(truth,predicted)),('Precision',precision_score(truth,predicted,zero_division=0)),('Recall',recall_score(truth,predicted,zero_division=0)),('F1',f1_score(truth,predicted,zero_division=0)),('PR-AUC',average_precision_score(truth,t.Predicted_Risk) if truth.nunique()>1 else float('nan'))]
    for col,(label,value) in zip(cols,metrics): col.metric(label, f'{value:.3f}')
    st.warning('Accuracy alone can mislead when failures are rare. Review precision, recall, PR-AUC and false alarms. Changing the threshold on this test view does not constitute independent validation.')
    cm = confusion_matrix(truth,predicted,labels=[0,1])
    st.plotly_chart(px.imshow(cm,text_auto=True,x=['No failure alert','Failure alert'],y=['No failure','Failure'],title='2025 test confusion matrix'),use_container_width=True)
    st.dataframe(latest[['Date','Machine','Risk (%)','Alert']].sort_values('Risk (%)',ascending=False),hide_index=True,use_container_width=True)

with pages[2]:
    st.subheader('Component status and last repair')
    latest_components = c.sort_values('Date').groupby(['Machine','Component']).tail(1).copy()
    last_repair = m[m.Maintenance_Type.str.contains('Corrective',case=False,na=False)].groupby(['Machine','Component']).Date.max().rename('Last_Corrective_Repair')
    last_service = m.groupby(['Machine','Component']).Date.max().rename('Last_Maintenance')
    latest_components = latest_components.join(last_repair,on=['Machine','Component']).join(last_service,on=['Machine','Component'])
    latest_components['Status'] = np.select([latest_components.Estimated_Health_Score < .40,latest_components.Estimated_Health_Score < .70],['Critical','Monitor'],default='Healthy')
    st.caption('Status thresholds are illustrative: Critical <40% health, Monitor 40–70%, Healthy ≥70%. Missing repair dates mean no corrective repair was recorded in the supplied history.')
    st.dataframe(latest_components[['Date','Machine','Component','Status','Estimated_Health_Score','Wear_Percentage','Days_Since_Last_Maintenance','Last_Corrective_Repair','Last_Maintenance']].sort_values(['Status','Wear_Percentage'],ascending=[True,False]),hide_index=True,use_container_width=True)
    st.plotly_chart(px.bar(latest_components,x='Component',y='Wear_Percentage',color='Status',hover_data=['Machine'],title='Latest component wear (%)'),use_container_width=True)
    st.plotly_chart(px.line(c.groupby('Date',as_index=False).Estimated_Health_Score.mean(),x='Date',y='Estimated_Health_Score',title='Historical mean health score'),use_container_width=True)

with pages[3]:
    st.subheader('Mean time between failures and mean time to repair')
    stats = p.groupby('Machine',as_index=False).agg(Operating_Hours=('Operating_Hours','sum'),Downtime_Hours=('Downtime_Hours','sum'),Reported_Failures=('Failure_Count','sum'))
    # MTBF = operating hours / count of recorded failures, an exposure-based historical estimate.
    failcounts = f.groupby('Machine').size().rename('Logged_Failures')
    stats = stats.join(failcounts,on='Machine').fillna({'Logged_Failures':0})
    stats['MTBF_Hours'] = stats.Operating_Hours / stats.Logged_Failures.replace(0,np.nan)
    # Repair duration from corrective maintenance entries only.
    corrective = m[m.Maintenance_Type.str.contains('Corrective',case=False,na=False)]
    repair_mean = corrective.groupby('Machine').Repair_Duration_Hours.mean().rename('MTTR_Hours')
    stats = stats.join(repair_mean,on='Machine')
    stats['Availability_Proxy'] = stats.Operating_Hours / (stats.Operating_Hours + stats.Downtime_Hours).replace(0,np.nan)
    st.caption('MTBF = total recorded operating hours / logged failure events. MTTR = mean recorded corrective repair duration (not necessarily all restoration time). Missing MTBF means zero logged failures; missing MTTR means no corrective repair records. Availability is a proxy, not audited OEE.')
    st.dataframe(stats.round(2),hide_index=True,use_container_width=True)
    if stats.MTBF_Hours.notna().any(): st.plotly_chart(px.bar(stats,x='Machine',y='MTBF_Hours',title='Estimated MTBF (operating hours)'),use_container_width=True)
    if stats.MTTR_Hours.notna().any(): st.plotly_chart(px.bar(stats,x='Machine',y='MTTR_Hours',title='Observed mean corrective repair time (hours)'),use_container_width=True)
    st.plotly_chart(px.bar(stats,x='Machine',y='Downtime_Hours',title='Recorded downtime (hours)'),use_container_width=True)

with pages[4]:
    st.subheader('Survival analysis and component replacement economics')
    st.info('Kaplan–Meier curves use first recorded failure per machine/component and right-censor components without a failure. With one observed component instance per machine/component and sparse events, component-specific probabilities are highly uncertain. Results are illustrative, not engineering life estimates.')
    machine_choice = st.selectbox('Survival cohort machine', machines, index=machines.index(selected) if selected in machines else 0)
    options = sorted(cohorts.loc[cohorts.Machine==machine_choice,'Component'].unique().tolist())
    component_choice = st.selectbox('Component for cost analysis',options)
    cohort = cohorts[cohorts.Component==component_choice]
    km = km_curve(cohort)
    n_events = int(cohort.Event.sum())
    st.caption(f'Pooled component type: {component_choice}; {len(cohort)} machine/component histories, {n_events} first failures. This pools machines and may hide different operating conditions.')
    st.plotly_chart(px.line(km,x='Day',y='Survival',line_shape='hv',title='Kaplan–Meier estimated survival since first observation',range_y=[0,1.02]),use_container_width=True)
    st.markdown('#### Scenario: remove now versus keep operating')
    col1,col2,col3 = st.columns(3)
    with col1:
        planned_cost = st.number_input('Planned removal + replacement cost (currency units)',min_value=0.0,value=1500.0,step=100.0)
        failure_cost = st.number_input('Unplanned failure consequence (repair + downtime)',min_value=0.0,value=5000.0,step=100.0)
    with col2:
        horizon = st.slider('Decision horizon (days)',7,365,90,7)
        daily_running_cost = st.number_input('Extra daily cost of keeping old component',min_value=0.0,value=0.0,step=10.0)
    with col3:
        replacement_residual_cost = st.number_input('Expected replacement failure cost over horizon',min_value=0.0,value=0.0,step=100.0)
        st.caption('All costs use the same currency. Enter local values (e.g., BWP). No cost inputs are contained in the dataset.')
    # Conditional survival S(age+h)/S(age) from empirical KM; not age-adjusted for covariates.
    component_record = components[(components.Machine==machine_choice)&(components.Component==component_choice)].sort_values('Date').tail(1)
    observed_age = int((component_record.Date.iloc[0]-components[(components.Machine==machine_choice)&(components.Component==component_choice)].Date.min()).days)
    def s_at(day):
        subset = km.loc[km.Day<=day,'Survival']
        return float(subset.iloc[-1]) if not subset.empty else 1.0
    base_survival = s_at(observed_age)
    conditional_failure = 1 - s_at(observed_age+horizon)/base_survival if base_survival>0 else float('nan')
    if n_events == 0 or np.isnan(conditional_failure):
        st.warning('Insufficient observed failures for an empirical conditional failure estimate. Entering cost inputs cannot create a defensible replacement recommendation.')
    else:
        keep_cost = conditional_failure * failure_cost + horizon * daily_running_cost
        remove_cost = planned_cost + replacement_residual_cost
        a,b,d = st.columns(3)
        a.metric('Estimated failure chance over horizon',f'{conditional_failure:.1%}')
        b.metric('Expected keep cost',f'{keep_cost:,.0f}')
        d.metric('Remove now cost',f'{remove_cost:,.0f}')
        st.plotly_chart(px.bar(pd.DataFrame({'Decision':['Keep until horizon','Remove now'],'Expected cost':[keep_cost,remove_cost]}),x='Decision',y='Expected cost',title='Scenario expected cost comparison'),use_container_width=True)
        st.write('Lower modeled cost: **'+('Remove now' if remove_cost<keep_cost else 'Keep operating')+'** — illustrative only; do not use as an automatic work order.')
    st.caption('Cost equation: keep = conditional failure probability × unplanned failure cost + horizon × daily extra running cost; remove = planned replacement cost + assumed replacement risk cost. It excludes discounting, multiple failures, opportunity costs, spare lead times and uncertain future maintenance.')

with pages[5]:
    a,b = st.columns(2)
    a.metric('Failure events',len(f));b.metric('Maintenance actions',len(m))
    if not f.empty:
        counts = f.Failure_Mode.value_counts().rename_axis('Failure_Mode').reset_index(name='Count')
        st.plotly_chart(px.bar(counts,x='Failure_Mode',y='Count',title='Failure mode frequency'),use_container_width=True)
    if not m.empty:
        st.plotly_chart(px.pie(m,names='Maintenance_Type',title='Maintenance type distribution'),use_container_width=True)
        st.dataframe(m.sort_values('Date',ascending=False).head(200),hide_index=True,use_container_width=True)

st.divider()
st.caption('Research prototype. Synthetic data; validate labels, calibration, event times, repair completeness, component cohorts, and costs before real maintenance decisions.')

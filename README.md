# Predictive Maintenance Digital Twin — Version 3

A Streamlit research demonstration based on **synthetic historical** maintenance datasets. Includes:

1. **Digital Twin Overview** — machine selection, component condition map, trends and alerts.
2. **Reliability** — MTBF, MTTR, availability proxy, failures, last repair and maintenance history.
3. **Survival & Cost** — Kaplan–Meier first-failure curves and scenario-based replacement cost estimates (illustrative, not validated).
4. **Manual Data Recording Station** — secure operator login, inspection, failure, repair and replacement forms, permanent Supabase storage and CSV export.

## Deploy to Streamlit Community Cloud

1. Extract this ZIP. Upload **the contents** of the `predictive_maintenance_v3` folder to the **root** of your existing GitHub repository, replacing `app.py`, `requirements.txt`, and `README.md`. Keep the `data` directory intact. Also upload `supabase_setup.sql` and `.gitignore` (the latter may be hidden on some systems).
2. Streamlit Community Cloud redeploys automatically. Main file path: `app.py`.
3. In Supabase SQL Editor, run `supabase_setup.sql` **after** creating the four tables (`inspections`, `failures`, `repairs`, `replacements`) from the earlier setup. If the tables do not exist, create them first. This script adds `user_id`, enables RLS, and creates policies for each authenticated user's own records.
4. In Supabase → Authentication → Users, add an operator account (email/password). Use the **operator** credentials in the Streamlit recording station, **not** the Supabase database password.
5. In Streamlit Cloud → Manage app → Settings → Secrets, paste:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_ANON_KEY = "YOUR_PUBLISHABLE_OR_ANON_KEY"
```

Get the URL and public/publishable key from Supabase → Project Settings → API Keys. **Never use a service_role/secret key in this app** and never commit real secrets to GitHub. If using a newer publishable key, confirm it works with your project's Supabase client.

## Important boundaries

- The historical demo plots do **not** automatically merge new Supabase records. New records appear in the recording station's Saved Records table immediately; integrating live data into validated MTBF/MTTR and model inference requires event matching, consistent operating-hour definitions and data-quality checks.
- RLS restricts each operator to **their own records**. For a shared team dashboard, configure roles and team-level policies separately.
- No live sensor readings are used. Model results, survival curves, and replacement-cost scenarios are not validated maintenance instructions.
- Do not publish confidential production records to a public GitHub repository.

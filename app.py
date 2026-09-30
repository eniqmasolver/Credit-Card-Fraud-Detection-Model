import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Fraud Shield | Transaction Check",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { --ink:#f4f7ff; --muted:#99a7c2; --blue:#4778ff; --line:#26334d; }
    .stApp { background:radial-gradient(ellipse at 78% 0%,#142448 0%,#0a1020 38%,#080d18 75%); color:var(--ink); }
    .block-container { max-width:1180px; padding-top:1.6rem; padding-bottom:3rem; }
    html, body, [class*="css"] { font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color:var(--ink); }
    .hero { display:flex; align-items:center; gap:16px; margin:1.4rem 0 1.8rem; }
    .topbar { display:flex; justify-content:space-between; align-items:center; padding:0 0 15px;
      border-bottom:1px solid rgba(81,101,140,.24); }
    .brand { color:#fff; font-size:13px; font-weight:800; letter-spacing:.13em; }
    .brand span { color:#7398ff; }
    .status { color:#9eb0cf; font-size:11px; }
    .status i { display:inline-block; width:7px; height:7px; border-radius:50%; background:#39cf91;
      margin:0 7px 1px 0; box-shadow:0 0 10px rgba(57,207,145,.6); }
    .hero-icon { width:54px; height:54px; border:1px solid #294276; border-radius:16px; display:grid;
      place-items:center; font-size:27px; background:linear-gradient(145deg,#172c58,#101a32); }
    .eyebrow { color:#83a4ff; text-transform:uppercase; letter-spacing:.16em; font-size:10px;
      font-weight:800; margin-bottom:7px; }
    .hero-title { font:800 34px ui-sans-serif,system-ui,sans-serif; letter-spacing:-.045em; margin:0; color:#fff; }
    .hero-sub { color:var(--muted); font-size:14px; margin:6px 0 0; }
    .panel-title { font-weight:750; font-size:18px; margin:0 0 5px; }
    .panel-copy { color:var(--muted); font-size:13px; margin:0 0 16px; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:rgba(14,22,39,.88); border-color:var(--line)!important;
      border-radius:18px!important; box-shadow:0 18px 55px rgba(0,0,0,.2); }
    .pill { display:inline-block; border:1px solid #2a4173; background:#111e39; color:#9bb5ff;
      padding:5px 10px; border-radius:99px; font-size:11px; font-weight:700; margin:9px 6px 8px 0; }
    .side-note { background:#0b1425; border:1px solid var(--line); border-radius:13px; padding:16px;
      margin:12px 0; }
    .side-note strong { display:block; font-weight:700; font-size:13px; margin-bottom:5px; color:#edf2ff; }
    .side-note span { color:var(--muted); font-size:12px; line-height:1.6; }
    div[data-testid="stTextArea"] textarea { border-radius:11px; background:#0a1221; color:#e6edff;
      border-color:#2a3852; font-size:13px; }
    div[data-testid="stTextArea"] label, div[data-testid="stCaptionContainer"] { color:var(--muted); }
    div.stButton > button, div[data-testid="stFormSubmitButton"] button { border-radius:10px;
      font-weight:700; min-height:42px; border-color:#2a3852; background:#111b2f; color:#e8eeff; }
    div.stButton > button:hover { border-color:#638cff; color:#fff; background:#172849; }
    div[data-testid="stFormSubmitButton"] button { background:linear-gradient(100deg,#3569f6,#5684ff);
      color:#fff; border:0; box-shadow:0 7px 22px rgba(55,105,246,.24); }
    div[data-testid="stFormSubmitButton"] button:hover { background:#5684ff; color:#fff; }
    [data-testid="stAlert"] { border-radius:12px; }
    [data-testid="stExpander"] { border-color:var(--line); border-radius:12px; background:#0b1425; }
    [data-testid="stExpander"] summary { color:#dbe5ff; }
    code { color:#a9c0ff!important; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as model_file:
        return pickle.load(model_file)


@st.cache_data(show_spinner="Loading transaction dataset…")
def load_dataset():
    dataset_path = Path(__file__).parent / "data" / "sample_transactions.csv"
    return pd.read_csv(dataset_path)


model = load_model()
feature_names = getattr(model, "feature_names_in_", None)
feature_count = model.n_features_in_
expected_features = (
    ", ".join(str(name) for name in feature_names)
    if feature_names is not None
    else f"{feature_count} values in the model's training order"
)
DATASET_AMOUNT_MEAN = 88.34961925093133
DATASET_AMOUNT_STD = 250.119670135235

NORMAL_SAMPLE = (
    "-1.359807134, -0.072781173, 2.536346738, 1.378155224, -0.338320770, "
    "0.462387778, 0.239598554, 0.098697901, 0.363786970, 0.090794172, "
    "-0.551599533, -0.617800856, -0.991389847, -0.311169354, 1.468176972, "
    "-0.470400525, 0.207971242, 0.025790580, 0.403992960, 0.251412098, "
    "-0.018306778, 0.277837576, -0.110473910, 0.066928075, 0.128539358, "
    "-0.189114844, 0.133558377, -0.021053053, 0.244964263370174"
)
FRAUD_SAMPLE = (
    "-2.312226542, 1.951992011, -1.609850732, 3.997905588, -0.522187865, "
    "-1.426545319, -2.537387306, 1.391657248, -2.770089277, -2.772272145, "
    "3.202033207, -2.899907388, -0.595221881, -4.289253782, 0.389724120, "
    "-1.140747180, -2.830055675, -0.016822468, 0.416955705, 0.126910559, "
    "0.517232371, -0.035049369, -0.465211076, 0.320198199, 0.044519167, "
    "0.177839798, 0.261145003, -0.143275874, -0.353229392966824"
)


def set_sample(value):
    st.session_state.transaction_features = value
    st.session_state.dataset_selected_class = None


st.markdown(
    '<div class="topbar"><div class="brand">FRAUD<span>/</span>LAB</div>'
    '<div class="status"><i></i>MODEL READY</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-icon">🛡️</div>
      <div>
        <div class="eyebrow">CREDIT RISK • MODEL WORKSPACE</div>
        <h1 class="hero-title">Transaction screening</h1>
        <p class="hero-sub">Check a transaction against the trained fraud detection model.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.65, 1], gap="large")

with left:
  with st.container(border=True):
    st.markdown('<p class="panel-title">Check a transaction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-copy">Paste the transaction features below, or load an example to explore the app.</p>',
        unsafe_allow_html=True,
    )
    with st.expander("Browse 100 sample Kaggle transactions", expanded=False):
        st.markdown(
            "This bundled sample contains **90 legitimate** and **10 fraud** transactions "
            "from the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)."
        )
        dataset = load_dataset()
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Sample rows", f"{len(dataset):,}")
        metric_b.metric("Fraud", f"{int((dataset['Class'] == 1).sum()):,}")
        metric_c.metric("Legitimate", f"{int((dataset['Class'] == 0).sum()):,}")
        class_choice = st.selectbox(
            "Browse transactions labeled",
            ["All", "Legitimate", "Fraud"],
            key="dataset_class_filter",
        )
        if class_choice == "Fraud":
            matching_rows = dataset.loc[dataset["Class"] == 1]
        elif class_choice == "Legitimate":
            matching_rows = dataset.loc[dataset["Class"] == 0]
        else:
            matching_rows = dataset

        row_number = st.number_input(
            f"Transaction row (1–{len(matching_rows)})",
            min_value=1,
            max_value=len(matching_rows),
            value=1,
            step=1,
            key=f"dataset_row_number_{class_choice}",
        )
        selected_row = matching_rows.iloc[int(row_number) - 1]
        st.dataframe(
            selected_row[["Time", "Amount", "Class", "V1", "V2", "V3"]]
            .rename("Value")
            .to_frame(),
            width="stretch",
        )
        if st.button("Use this transaction in the checker", width="stretch"):
            normalized_amount = (selected_row["Amount"] - DATASET_AMOUNT_MEAN) / DATASET_AMOUNT_STD
            transaction_values = [
                *[selected_row[f"V{i}"] for i in range(1, 29)],
                normalized_amount,
            ]
            st.session_state.transaction_features = ", ".join(
                f"{float(value):.12g}" for value in transaction_values
            )
            st.session_state.dataset_selected_class = int(selected_row["Class"])
            st.rerun()

    sample_a, sample_b, sample_c = st.columns([1, 1, 1.15])
    with sample_a:
        st.button("↗  Example input", width="stretch", on_click=set_sample, args=(NORMAL_SAMPLE,))
    with sample_b:
        st.button("⚑  Fraud example", width="stretch", on_click=set_sample, args=(FRAUD_SAMPLE,))
    with sample_c:
        st.button("Clear input", width="stretch", on_click=set_sample, args=("",))

    with st.form("transaction_check"):
        input_data = st.text_area(
            "Feature values",
            key="transaction_features",
            height=150,
            placeholder=f"Paste {feature_count} comma-separated numeric values here…",
            help="Values must follow the feature order shown in the panel to the right.",
        )
        entered_count = len([part for part in input_data.split(",") if part.strip()])
        st.caption(f"{entered_count} of {feature_count} values entered")
        submit = st.form_submit_button("Analyze transaction", width="stretch")
    

    if submit:
        if not input_data.strip():
            st.warning("Enter the transaction feature values first, or load an example.")
        else:
            try:
                values = [float(value.strip()) for value in input_data.split(",")]
            except ValueError:
                st.error("Every feature must be a valid number. Separate values with commas.")
            else:
                if len(values) != feature_count:
                    st.error(
                        f"You entered {len(values)} values; this model requires {feature_count}. "
                        "Check the feature order shown alongside the form."
                    )
                else:
                    try:
                        prediction = model.predict(np.asarray(values).reshape(1, -1))[0]
                    except Exception as exc:
                        st.error(f"The model could not make a prediction: {exc}")
                    else:
                        st.markdown("#### Analysis result")
                        if prediction == 0:
                            st.success("✅  No fraud detected for this transaction")
                        else:
                            st.error("⚠️  The model flagged this transaction as potentially fraudulent")
                        dataset_label = st.session_state.get("dataset_selected_class")
                        if dataset_label is not None:
                            label_text = "Fraud" if dataset_label == 1 else "Legitimate"
                            st.caption(f"Kaggle dataset label for this transaction: {label_text}")

with right:
    st.markdown('<div class="panel-title">Input guide</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="pill">{feature_count} features</span><span class="pill">Comma separated</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-note"><strong>Feature order matters</strong>'
        '<span>The model expects V1–V28 followed by Normalized_Amount. Keep the values in this exact order.</span></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Show the full feature sequence"):
        st.code(expected_features, language=None)
    st.markdown(
        '<div class="side-note"><strong>What are V1–V28?</strong>'
        '<span>These are anonymized PCA features from the source dataset. Use actual transaction feature values or the examples above.</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("For demonstration and learning. Predictions are not a guarantee of transaction legitimacy.")

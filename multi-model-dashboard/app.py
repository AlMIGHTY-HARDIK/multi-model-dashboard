import warnings
# Filter out warnings about Matplotlib GUI and SHAP guessing
warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread")
warnings.filterwarnings("ignore", message="Parameter shap='guess'")

# --- Monkey Patch Plotly's YAxis to remove the invalid "titlefont" property ---
import plotly.graph_objects as go
validators = go.layout.YAxis.__dict__.get("_validators")
if validators and isinstance(validators, dict) and "titlefont" in validators:
    go.layout.YAxis._validators = {k: v for k, v in validators.items() if k != "titlefont"}

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import io
import base64
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_curve, confusion_matrix
)

from explainerdashboard import ClassifierExplainer
from explainerdashboard.custom import (
    ExplainerComponent, ShapSummaryComponent, ImportancesComponent,
    ConfusionMatrixComponent, FeatureInputComponent, ShapDependenceComponent
)

import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, Dash
import plotly.express as px

##############################################################################
# Helper: Decide SHAP Explainer Type
##############################################################################
def decide_shap_type(model_key):
    if model_key in ["rf", "dt"]:
        return "tree"
    else:
        return "kernel"

##############################################################################
# Helper: Create a Precision-Recall Plot
##############################################################################
def create_precision_recall_plot(explainer, label):
    y_test = explainer.y_test if hasattr(explainer, "y_test") else explainer.y
    X_test = explainer.X_test if hasattr(explainer, "X_test") else explainer.X
    prec_vals, rec_vals, _ = precision_recall_curve(
        y_test, explainer.model.predict_proba(X_test)[:, 1]
    )
    fig, ax = plt.subplots()
    ax.plot(rec_vals, prec_vals, marker='.', label=label)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

##############################################################################
# Custom Dashboard Component
##############################################################################
class CustomDashboard(ExplainerComponent):
    def __init__(self, explainer, prefix="", name=None):
        super().__init__(explainer, name=name)
        self.prefix = prefix
        self.shap_summary = ShapSummaryComponent(explainer)
        self.feature_importance = ImportancesComponent(explainer)
        self.confusion_matrix = ConfusionMatrixComponent(explainer)
        self.feature_input = FeatureInputComponent(explainer)
        X_data = explainer.X_test if hasattr(explainer, "X_test") else explainer.X
        num_cols = X_data.select_dtypes(exclude=["object"]).columns.tolist()
        dep_col = num_cols[0] if len(num_cols) > 0 else None
        self.dependence = ShapDependenceComponent(explainer, col=dep_col) if dep_col else None
        self.pr_curve_img = create_precision_recall_plot(explainer, label=name or "Model")

    def layout(self):
        y_true = self.explainer.y_test if hasattr(self.explainer, "y_test") else self.explainer.y
        X_data = self.explainer.X_test if hasattr(self.explainer, "X_test") else self.explainer.X
        cm = confusion_matrix(y_true, self.explainer.model.predict(X_data))
        print(f"Confusion Matrix for {self.name}: \n{cm}")
        components = [
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(html.H4("Model Overview, Inputs, and Prediction", className="text-primary")),
                        dbc.CardBody([
                            html.H5("Model Metrics", className="mt-3"),
                            html.Ul([
                                html.Li(f"Accuracy: {accuracy_score(y_true, self.explainer.model.predict(X_data)):.2%}"),
                                html.Li(f"F1 Score: {f1_score(y_true, self.explainer.model.predict(X_data)):.2%}"),
                                html.Li(f"Precision: {precision_score(y_true, self.explainer.model.predict(X_data)):.2%}"),
                                html.Li(f"Recall: {recall_score(y_true, self.explainer.model.predict(X_data)):.2%}")
                            ], style={'fontSize': '1rem'}),
                            html.Label("Input Feature Example:", style={'fontWeight': 'bold'}),
                            dcc.Input(
                                id=f"{self.prefix}input-feature",
                                type="text",
                                placeholder="Enter value",
                                style={'marginBottom': '15px'}
                            ),
                            html.Button("Predict", id=f"{self.prefix}predict-button", n_clicks=0, className="btn btn-primary"),
                            html.H5("Prediction Result:", className="mt-4"),
                            html.Div(
                                id=f"{self.prefix}prediction-output",
                                className="text-success font-weight-bold",
                                style={'fontSize': '1.1rem'}
                            )
                        ])
                    ], className="shadow-sm animate__animated animate__fadeIn", 
                       style={'borderRadius': '15px', 'border': '1px solid #007bff', 'padding': '10px'}),
                    width=6
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(html.H4("Explainable AI Insights", className="text-primary")),
                        dbc.CardBody([
                            html.H5("SHAP Summary Plot", className="mt-3"),
                            self.shap_summary.layout(),
                            html.H5("Feature Importance", className="mt-4"),
                            self.feature_importance.layout(),
                            html.H5("Dependence Plot", className="mt-4"),
                            self.dependence.layout() if self.dependence else html.P("Not available"),
                            html.H5("Confusion Matrix", className="mt-4"),
                            self.confusion_matrix.layout(),
                            html.H5("Precision-Recall Curve", className="mt-4"),
                            html.Img(src=f"data:image/png;base64,{self.pr_curve_img}",
                                     style={'width': '100%', 'marginBottom': '20px'})
                        ])
                    ], className="shadow-sm animate__animated animate__fadeIn", 
                       style={'borderRadius': '15px', 'border': '1px solid #28a745', 'padding': '10px'}),
                    width=6
                )
            ]),
            dbc.Row([
                dbc.Col(
                    html.Footer("© 2025 Prediction Dashboard | Data-Driven Insights & Explainability",
                                className="text-center text-light bg-dark p-3 mt-4",
                                style={'fontFamily': 'Courier New, monospace', 'fontSize': '1rem'})
                )
            ])
        ]
        return dbc.Container(components, fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px'})

##############################################################################
# Data Loading & Preprocessing
##############################################################################
def load_dataset(dataset_key):
    mapping = {
        "heart":  r"saved_datasets\Heart_UCI.csv",
        "house":  r"saved_datasets\House_Price_Prediction.csv",
        "loan":   r"saved_datasets\Loan_Prediction.csv",
        "stocks": r"saved_datasets\Stocks.csv",
        "adult":  r"saved_datasets\UCI_Adult_Census_Income.csv"
    }
    if dataset_key not in mapping:
        raise ValueError("Unknown dataset key.")
    df = pd.read_csv(mapping[dataset_key])
    return df

def preprocess_dataset(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_cat = encoder.fit_transform(X[cat_cols])
        X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols))
    else:
        X_cat_df = pd.DataFrame()
    X_num_df = X[num_cols].reset_index(drop=True)
    X_processed = pd.concat([X_num_df, X_cat_df.reset_index(drop=True)], axis=1)
    return X_processed, y

def get_model(model_key):
    if model_key == "svm":
        return SVC(probability=True, random_state=42)
    elif model_key == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_key == "dt":
        return DecisionTreeClassifier(random_state=42)
    elif model_key == "knn":
        return KNeighborsClassifier()
    elif model_key == "nb":
        return GaussianNB()
    else:
        raise ValueError("Unknown model key.")

##############################################################################
# Main Dash App Layout
##############################################################################
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    # If you haven't already, include Animate.css for animations
    "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"
]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# "Story" or Intro Card at the top
# "Story" or Intro Card at the top with enhanced design
story_header = dbc.Container(
    dbc.Card([
        dbc.CardHeader(
            html.H2("FACTS-H Dashboard", className="m-0 text-white"),
            className="bg-success text-center",
            style={"fontWeight": "bold", "fontSize": "2rem", "padding": "20px"}
        ),
        dbc.CardBody([
            # Introductory text above the snapshots
            html.P(
                "Rahul and Priya, two ambitious professionals, are trying to buy their dream home. "
                "They walk into the same bank with identical financial profiles and are excited to get their loan approved.",
                className="lead text-center",
                style={"fontSize": "1.25rem", "marginBottom": "30px"}
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H4("💰 Rahul’s Financial Snapshot", className="text-white"),
                            className="bg-success",
                            style={"padding": "10px"}
                        ),
                        dbc.CardBody(
                            html.Ul([
                                html.Li("Salary: ₹80,000/month"),
                                html.Li("Credit Score: 750"),
                                html.Li("Work Experience: 5 Years")
                            ], style={"fontSize": "1rem", "margin": "0"})
                        )
                    ], className="mb-3 shadow-sm")
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H4("👩‍💼 Priya’s Financial Snapshot", className="text-white"),
                            className="bg-success",
                            style={"padding": "10px"}
                        ),
                        dbc.CardBody(
                            html.Ul([
                                html.Li("Salary: ₹80,000/month"),
                                html.Li("Credit Score: 750"),
                                html.Li("Work Experience: 5 Years")
                            ], style={"fontSize": "1rem", "margin": "0"})
                        )
                    ], className="mb-3 shadow-sm")
                ], width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H5("🚨 The AI Shock: A Decision Without a Reason!", className="text-danger text-center", style={"fontWeight": "bold"}),
                        html.P([
                            "✅ Rahul gets approved! ",
                            html.Span("🎉", className="text-success"),
                            html.Br(),
                            "❌ Priya gets rejected! ",
                            html.Span("😲", className="text-danger")
                        ], className="text-center", style={"fontSize": "1.25rem", "marginTop": "10px"})
                    ], style={"padding": "10px", "border": "1px solid #e0e0e0", "borderRadius": "8px"})
                )
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.P("Priya is stunned. “Wait… we have the SAME financial history! What went wrong?” "
                               "She rushes to the bank manager:"),
                        html.P("🗣️ “Why was I rejected?” The manager shrugs:"),
                        html.P("💻 “Our AI model made the decision… but we don’t know why.”")
                    ], className="text-center", style={"fontSize": "1rem"})
                )
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4("A powerful system making life-changing decisions, but with zero transparency?", className="text-warning text-center"),
                        html.P("That’s when Explainable AI (XAI) steps in!", className="text-center", style={"fontSize": "1.1rem"})
                    ], style={"padding": "10px"})
                )
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H5("🕵️‍♂️ The Investigation: XAI Exposes AI’s Hidden Flaws!", className="text-danger text-center"),
                        html.P([
                            "🚨 Hidden Bias Detected! The AI model was trained on historical loan approvals, where men were more likely to get loans. ",
                            "Priya wasn’t rejected because of her finances—she was denied due to a flawed AI decision! ",
                            "🤯 AI wasn’t trying to be unfair, but without explainability, no one knew why it was making biased choices!"
                        ], className="text-center", style={"fontSize": "1rem"})
                    ], style={"padding": "10px", "backgroundColor": "#f8d7da", "borderRadius": "8px"})
                )
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H5("🛠️ XAI to the Rescue: Bringing AI Transparency!", className="text-primary text-center"),
                        html.P([
                            "✅ Decode AI Decisions – No more “mystery machine” AI!",
                            html.Br(),
                            "✅ Expose Hidden Biases – AI models can be trained better when bias is detected early.",
                            html.Br(),
                            "✅ Ensure Fairness & Trust – AI now provides clear justifications for every approval or rejection.",
                            html.Br(),
                            "📊 Re-evaluating Priya’s Application… 🎉 Priya’s Loan Status: ✅ Approved!"
                        ], className="text-center", style={"fontSize": "1rem"})
                    ], style={"padding": "10px", "backgroundColor": "#d1ecf1", "borderRadius": "8px"})
                )
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H5("🌟 What’s the Lesson Here?", className="text-info text-center"),
                        html.P([
                            "💭 Would you trust an AI model that can’t explain itself?",
                            html.Br(),
                            "✔ AI without explainability is a black box—dangerous & unpredictable.",
                            html.Br(),
                            "✔ Bias can exist in AI models—even if unintended.",
                            html.Br(),
                            "✔ With Explainable AI, decisions aren’t just made—they’re justified!"
                        ], className="text-center", style={"fontSize": "1rem"})
                    ], style={"padding": "10px"})
                )
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.P("🔍 Ready to See What AI Is Thinking?", className="text-center", style={"fontSize": "1.2rem", "fontWeight": "bold"}),
                        html.P("💡 Want to unlock the WHY behind AI decisions?", className="text-center", style={"fontSize": "1.1rem"}),
                        html.P(html.Span("🔎 Click “Check AI Explainability” and reveal the truth!", className="font-weight-bold text-primary"), className="text-center")
                    ], style={"padding": "10px"})
                )
            ], className="mb-0")
        ], className="animate__animated animate__fadeIn")
    ], className="shadow-sm border-0"),
    fluid=True, style={'marginTop': '40px', 'marginBottom': '40px'}
)


DATASET_OPTIONS = [
    {"label": "Heart_UCI.csv", "value": "heart"},
    {"label": "House_Price_Prediction.csv", "value": "house"},
    {"label": "Loan_Prediction.csv", "value": "loan"},
    {"label": "Stocks.csv", "value": "stocks"},
    {"label": "UCI_Adult_Census_Income.csv", "value": "adult"},
]

MODEL_OPTIONS = [
    {"label": "SVM", "value": "svm"},
    {"label": "Random Forest", "value": "rf"},
    {"label": "Decision Tree", "value": "dt"},
    {"label": "KNN", "value": "knn"},
    {"label": "Naive Bayes", "value": "nb"},
]

app.layout = dbc.Container(fluid=True, children=[
    # The "story" card at the top
    story_header,

    dbc.Row([
        dbc.Col([
            html.H4("Select Your Dataset"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=DATASET_OPTIONS,
                placeholder="Choose a dataset",
                clearable=False
            )
        ], width=6),
        dbc.Col([
            html.H4("Select a Model"),
            dcc.Dropdown(
                id="model-dropdown",
                options=MODEL_OPTIONS,
                placeholder="Choose a model",
                clearable=False
            )
        ], width=6)
    ], className="mt-4"),

    dbc.Row([
        dbc.Col([
            html.H5("AI Decisions: Before and After Explainability"),
            dcc.Graph(id="decisions-graph", figure={})
        ], width=6),
        dbc.Col([
            html.H5("Bias Check: Before and After Explainability"),
            dcc.Graph(id="bias-graph", figure={})
        ], width=6)
    ], className="mt-4"),

    html.Hr(),

    html.Div(id="xai-dashboard-output")
])

##############################################################################
# Callback: Update Charts & Embed Custom Dashboard Layout
##############################################################################
@app.callback(
    Output("decisions-graph", "figure"),
    Output("bias-graph", "figure"),
    Output("xai-dashboard-output", "children"),
    Input("dataset-dropdown", "value"),
    Input("model-dropdown", "value")
)
def update_interface(selected_dataset, selected_model):
    if not selected_dataset or not selected_model:
        return {}, {}, ""

    try:
        df = load_dataset(selected_dataset)
    except Exception as e:
        return {}, {}, dbc.Alert(f"Error loading dataset: {e}", color="danger")

    try:
        X, y = preprocess_dataset(df)
    except Exception as e:
        return {}, {}, dbc.Alert(f"Error preprocessing dataset: {e}", color="danger")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if len(X_test) == 0:
        return {}, {}, dbc.Alert("No test data after splitting. Check dataset size.", color="danger")

    try:
        model = get_model(selected_model)
        model.fit(X_train, y_train)
    except Exception as e:
        return {}, {}, dbc.Alert(f"Error training model: {e}", color="danger")

    shap_type = decide_shap_type(selected_model)
    bg_size = min(50, len(X_test))
    bg = X_test.sample(bg_size, random_state=42)

    explainer = ClassifierExplainer(
        model, X_test, y_test,
        shap=shap_type,
        X_background=bg
    )
    explainer.calculate_properties()

    # Debug: Print sample SHAP values
    df_shap = explainer.get_shap_values_df()
    print(f"--- SHAP Values Sample for {selected_dataset.upper()} + {selected_model.upper()} ---")
    print(df_shap.head(10))

    dashboard = CustomDashboard(
        explainer, prefix=f"{selected_dataset}_{selected_model}_",
        name=f"{selected_dataset.upper()} + {selected_model.upper()}"
    )

    # Dummy Bar Chart for Decisions
    decisions_data = {
        "Decision_1": [0.70, 0.65],
        "Decision_2": [0.80, 0.77],
        "Decision_3": [0.60, 0.58],
    }
    df_decisions = px.data.tips().head(3).copy()
    df_decisions["Decision"] = list(decisions_data.keys())
    df_decisions["Before"] = [v[0] for v in decisions_data.values()]
    df_decisions["After"]  = [v[1] for v in decisions_data.values()]
    fig_decisions = px.bar(
        df_decisions, x="Decision", y=["Before", "After"],
        barmode="group",
        title=f"Decisions for {selected_model.upper()}"
    )

    # Dummy Bar Chart for Bias
    bias_data = {
        "Gender Bias": [0.60, 0.40],
        "Income Bias": [0.70, 0.50],
        "Age Bias":    [0.80, 0.65],
    }
    df_bias = px.data.tips().head(3).copy()
    df_bias["Bias_Type"] = list(bias_data.keys())
    df_bias["Before"]    = [v[0] for v in bias_data.values()]
    df_bias["After"]     = [v[1] for v in bias_data.values()]
    fig_bias = px.bar(
        df_bias, x="Bias_Type", y=["Before", "After"],
        barmode="group",
        title=f"Bias Check for {selected_dataset.upper()}"
    )

    return fig_decisions, fig_bias, dashboard.layout()

##############################################################################
# Optional: FastAPI Server Setup
##############################################################################
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

server = FastAPI()
server.mount("/", WSGIMiddleware(app.server))

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

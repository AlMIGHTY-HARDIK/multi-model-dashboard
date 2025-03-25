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
from explainerdashboard.custom import ExplainerComponent

import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, Dash
import plotly.express as px
import shap

##############################################################################
# Decide SHAP Explainer Type
##############################################################################
def decide_shap_type(model_key):
    if model_key in ["rf", "dt"]:
        return "tree"
    else:
        return "kernel"

##############################################################################
# Create a Precision-Recall Plot
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
# Create Confusion Matrix Table
##############################################################################
def create_confusion_matrix_table(cm):
    """
    Returns a Dash HTML table displaying the confusion matrix with row/column headers.
    Expects cm to be a 2D list/array, e.g. [[tn, fp], [fn, tp]].
    """
    return html.Table([
        html.Thead(
            html.Tr([
                html.Th(""),
                html.Th("Predicted 0", style={"padding": "6px 12px"}),
                html.Th("Predicted 1", style={"padding": "6px 12px"})
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Th("Actual 0", style={"padding": "6px 12px"}),
                html.Td(str(cm[0][0]), style={"padding": "6px 12px"}),
                html.Td(str(cm[0][1]), style={"padding": "6px 12px"})
            ]),
            html.Tr([
                html.Th("Actual 1", style={"padding": "6px 12px"}),
                html.Td(str(cm[1][0]), style={"padding": "6px 12px"}),
                html.Td(str(cm[1][1]), style={"padding": "6px 12px"})
            ])
        ])
    ],
    style={
        "border": "1px solid #ddd",
        "borderCollapse": "collapse",
        "marginTop": "10px",
        "width": "100%"
    })

##############################################################################
# Custom Helper Functions for SHAP Components
##############################################################################
def get_custom_shap_summary(explainer, shap_type):
    """
    Generates a custom SHAP summary bar plot using Plotly.
    """
    X_test = explainer.X_test if hasattr(explainer, "X_test") else explainer.X
    bg = X_test.sample(min(50, len(X_test)), random_state=42)
    model = explainer.model

    if shap_type == "tree":
        explainer_shap = shap.TreeExplainer(model, data=bg)
        shap_values = explainer_shap.shap_values(X_test)
        shap_array = shap_values[1] if isinstance(shap_values, list) and len(shap_values)==2 else shap_values
    else:
        explainer_shap = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer_shap.shap_values(X_test)
        shap_array = shap_values[1] if isinstance(shap_values, list) and len(shap_values)==2 else shap_values

    if shap_array.ndim == 3 and shap_array.shape[2] == 2:
        shap_array = shap_array[:, :, 1]

    # Align columns if necessary
    n_shap_cols = shap_array.shape[1]
    n_actual_cols = X_test.shape[1]
    if n_shap_cols > n_actual_cols:
        shap_array = shap_array[:, :n_actual_cols]
    elif n_shap_cols < n_actual_cols:
        X_test = X_test.iloc[:, :n_shap_cols]

    feature_names = list(X_test.columns)
    mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
    df_shap = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    df_shap = df_shap.sort_values("mean_abs_shap", ascending=True)

    fig = px.bar(
        df_shap,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        labels={"mean_abs_shap": "Mean |SHAP value|", "feature": "Attribute"},
        title="Custom SHAP Summary (Bar Plot)"
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return dcc.Graph(figure=fig)

def get_custom_feature_importance(explainer, shap_type):
    """
    Generates a custom feature importance plot.
    Uses model.feature_importances_ if available; otherwise, falls back to SHAP mean abs values.
    """
    X_test = explainer.X_test if hasattr(explainer, "X_test") else explainer.X
    model = explainer.model

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = list(X_test.columns)
        if len(importances) > len(feature_names):
            importances = importances[:len(feature_names)]
        elif len(importances) < len(feature_names):
            feature_names = feature_names[:len(importances)]
    else:
        bg = X_test.sample(min(50, len(X_test)), random_state=42)
        if shap_type == "tree":
            explainer_shap = shap.TreeExplainer(model, data=bg)
            shap_values = explainer_shap.shap_values(X_test)
            shap_array = shap_values[1] if isinstance(shap_values, list) and len(shap_values)==2 else shap_values
        else:
            explainer_shap = shap.KernelExplainer(model.predict_proba, bg)
            shap_values = explainer_shap.shap_values(X_test)
            shap_array = shap_values[1] if isinstance(shap_values, list) and len(shap_values)==2 else shap_values

        if shap_array.ndim == 3 and shap_array.shape[2] == 2:
            shap_array = shap_array[:, :, 1]

        n_shap_cols = shap_array.shape[1]
        n_actual_cols = X_test.shape[1]
        if n_shap_cols > n_actual_cols:
            shap_array = shap_array[:, :n_actual_cols]
        elif n_shap_cols < n_actual_cols:
            X_test = X_test.iloc[:, :n_shap_cols]

        feature_names = list(X_test.columns)
        importances = np.mean(np.abs(shap_array), axis=0)

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=True)

    fig = px.bar(
        df_imp,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Importance", "feature": "Attribute"},
        title="Custom Feature Importance"
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return dcc.Graph(figure=fig)

def get_custom_dependence_plot(explainer, shap_type, selected_feature=None):
    """
    Generates a custom SHAP dependence plot.
    This function creates a matplotlib figure via shap.dependence_plot, then encodes it
    as a PNG image and embeds it in a Plotly figure.
    """
    X_test = explainer.X_test if hasattr(explainer, "X_test") else explainer.X
    model = explainer.model
    bg = X_test.sample(min(50, len(X_test)), random_state=42)

    if shap_type == "tree":
        explainer_shap = shap.TreeExplainer(model, data=bg)
        shap_values = explainer_shap.shap_values(X_test)
        shap_array = shap_values[1] if isinstance(shap_values, list) and len(shap_values)==2 else shap_values
    else:
        explainer_shap = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer_shap.shap_values(X_test)
        shap_array = shap_values[1] if isinstance(shap_values, list) and len(shap_values)==2 else shap_values

    if shap_array.ndim == 3 and shap_array.shape[2] == 2:
        shap_array = shap_array[:, :, 1]

    if selected_feature is None or selected_feature not in X_test.columns:
        selected_feature = X_test.columns[0]

    fig_dep = plt.figure()
    shap.dependence_plot(selected_feature, shap_array, X_test, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig_dep)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source="data:image/png;base64," + encoded,
            xref="paper", yref="paper",
            x=0, y=1,
            sizex=1, sizey=1,
            xanchor="left", yanchor="top"
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        width=700,
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Custom SHAP Dependence Plot: {selected_feature}"
    )
    return dcc.Graph(figure=fig)

##############################################################################
# Custom Dashboard Component
##############################################################################
class CustomDashboard(ExplainerComponent):
    def __init__(self, explainer, prefix="", name=None, model_key=None):
        super().__init__(explainer, name=name)
        self.prefix = prefix
        self.model_key = model_key
        self.shap_type = decide_shap_type(model_key) if model_key else "kernel"
        # Generate the precision-recall plot (base64 encoded)
        self.pr_curve_img = create_precision_recall_plot(explainer, label=name or "Model")

    def layout(self):
        y_true = self.explainer.y_test if hasattr(self.explainer, "y_test") else self.explainer.y
        X_data = self.explainer.X_test if hasattr(self.explainer, "X_test") else self.explainer.X
        cm = confusion_matrix(y_true, self.explainer.model.predict(X_data))

        # Left Card 1: Model Overview, Inputs, and Prediction
        left_overview = dbc.Card([
            dbc.CardHeader(html.H4("Model Overview, Inputs, and Prediction", className="text-primary")),
            dbc.CardBody([
                # Model Metrics Heading + Description
                html.H5("📈 Model Metrics (Accuracy, F1 Score, Precision, Recall)", className="mt-3"),
                html.P(
                    "These numbers indicate how well the AI model performs. A higher score means a more "
                    "reliable AI model. Metrics include:\n\n"
                    "• Accuracy: The overall correctness of the model.\n\n"
                    "• Precision: When the model predicts “approved,” how often is it correct?\n\n"
                    "• Recall: How well does the model catch all relevant cases?\n\n"
                    "• F1 Score: A balance between precision and recall.",
                    style={"textAlign": "justify", "fontSize": "0.9rem", "marginBottom": "15px"}
                ),
                html.Ul([
                    html.Li(f"Accuracy: {accuracy_score(y_true, self.explainer.model.predict(X_data)):.2%}"),
                    html.Li(f"F1 Score: {f1_score(y_true, self.explainer.model.predict(X_data)):.2%}"),
                    html.Li(f"Precision: {precision_score(y_true, self.explainer.model.predict(X_data)):.2%}"),
                    html.Li(f"Recall: {recall_score(y_true, self.explainer.model.predict(X_data)):.2%}")
                ], style={'fontSize': '1rem', 'marginBottom': '20px'}),

                # Input & Prediction
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
           style={'borderRadius': '15px', 'border': '1px solid #007bff', 'padding': '10px'})

        # Left Card 2: Confusion Matrix & Precision-Recall Curve
        left_additional = dbc.Card([
            dbc.CardHeader(html.H4("Confusion Matrix & Precision-Recall Curve", className="text-primary")),
            dbc.CardBody([
                # Confusion Matrix Heading + Description
                html.H5("🔍 Confusion Matrix", className="mt-3"),
                html.P(
                    "The confusion matrix helps in understanding AI’s prediction accuracy. "
                    "It shows where the model made correct predictions and where it made mistakes "
                    "(e.g., incorrectly rejecting or approving loans).",
                    style={"textAlign": "justify", "fontSize": "0.9rem", "marginBottom": "15px"}
                ),
                create_confusion_matrix_table(cm),

                # Precision-Recall Curve Heading + Description
                html.H5("📈 Precision-Recall Curve", className="mt-4"),
                html.P(
                    "This plot balances precision and recall in AI decision-making. If precision is too high, "
                    "the model may miss valid cases. If recall is too high, the model may approve incorrect cases. "
                    "This graph helps in optimizing AI decision thresholds.",
                    style={"textAlign": "justify", "fontSize": "0.9rem", "marginBottom": "15px"}
                ),
                html.Img(
                    src=f"data:image/png;base64,{self.pr_curve_img}",
                    style={'width': '100%', 'marginBottom': '20px'}
                )
            ])
        ], className="shadow-sm animate__animated animate__fadeIn", 
           style={'borderRadius': '15px', 'border': '1px solid #007bff', 'padding': '10px', 'marginTop': '20px'})

        left_column = dbc.Col([left_overview, left_additional], width=6)

        # Right Column: Custom Explainable AI Insights
        right_column = dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Explainable AI Insights (Custom)", className="text-primary")),
                dbc.CardBody([
                    # SHAP Summary Plot Heading + Description
                    html.H5("📊 SHAP Summary Plot (Feature Importance Analysis)", className="mt-3"),
                    html.P(
                        "This plot reveals which features (e.g., income, employment status) influence AI "
                        "decisions the most. Understanding which factors drive AI decisions ensures "
                        "transparency and allows for adjustments if needed.",
                        style={"textAlign": "justify", "fontSize": "0.9rem", "marginBottom": "15px"}
                    ),
                    get_custom_shap_summary(self.explainer, self.shap_type),

                    # Feature Importance
                    html.H5("Feature Importance", className="mt-4"),
                    html.P(
                        "Feature importance ranks features by their overall impact on the model’s predictions. "
                        "A higher bar indicates that the feature has a stronger influence. "
                        "This helps answer the question: Which inputs matter the most for deciding if someone is Vulnerable?",
                        style={"textAlign": "justify", "fontSize": "0.9rem", "marginBottom": "15px"}
                    ),
                    html.Ul([
                        html.Li("High importance suggests the feature heavily influences the final prediction."),
                        html.Li("Low importance indicates the feature plays a smaller role."),
                        html.Li("Comparing bars helps identify the top factors driving the model’s decisions."),
                    ], style={"fontSize": "0.9rem", "marginBottom": "15px"}),
                    get_custom_feature_importance(self.explainer, self.shap_type),

                    # SHAP Dependence Plot Heading + Description
                    html.H5("🔎 SHAP Dependence Plot", className="mt-4"),
                    html.P(
                        "This plot shows how changing a specific factor (e.g., self-employment status) "
                        "impacts AI decisions. It helps users see which inputs affect predictions the "
                        "most, making AI decision-making more interpretable.",
                        style={"textAlign": "justify", "fontSize": "0.9rem", "marginBottom": "15px"}
                    ),
                    get_custom_dependence_plot(self.explainer, self.shap_type)
                ])
            ], className="shadow-sm animate__animated animate__fadeIn", 
               style={'borderRadius': '15px', 'border': '1px solid #28a745', 'padding': '10px'}),
            width=6
        )

        # Combine columns
        components = [
            dbc.Row([left_column, right_column]),
            dbc.Row([
                dbc.Col(
                    html.Footer(
                        "© 2025 Prediction Dashboard | Data-Driven Insights & Explainability",
                        className="text-center text-light bg-dark p-3 mt-4",
                        style={'fontFamily': 'Courier New, monospace', 'fontSize': '1rem'}
                    )
                )
            ])
        ]
        return dbc.Container(components, fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px'})

##############################################################################
# Data Loading & Preprocessing
##############################################################################
def load_dataset(dataset_key):
    mapping = {
        "heart":  r"saved_datasets/Heart_UCI.csv",
        "house":  r"saved_datasets/House_Price_Prediction.csv",
        "loan":   r"saved_datasets/Loan_Prediction.csv",
        "stocks": r"saved_datasets/Stocks.csv",
        "adult":  r"saved_datasets/UCI_Adult_Census_Income.csv"
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
            html.H5("📊 AI Decisions: Before and After Explainability"),
            html.P(
                "This plot illustrates how AI decisions change after applying explainability "
                "techniques. Before explainability, AI made decisions without transparency. "
                "After applying explainability tools, biases were identified and adjusted, "
                "making AI decisions more trustworthy and fair.",
                style={"textAlign": "justify", "fontSize": "0.9rem", "marginBottom": "15px"}
            ),
            dcc.Graph(id="decisions-graph", figure={})
        ], width=6),
        dbc.Col([
            html.H5("🚨 Bias Check: Before and After Explainability"),
            html.P(
                "This visualization highlights the biases in AI decision-making before and "
                "after explainability techniques were applied. Initially, the AI model might "
                "favor certain groups (e.g., based on gender or income). With explainability, "
                "these biases are detected and corrected, ensuring fairness in decision-making.",
                style={"textAlign": "justify", "fontSize": "0.9rem", "marginBottom": "15px"}
            ),
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
        name=f"{selected_dataset.upper()} + {selected_model.upper()}",
        model_key=selected_model
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

import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host="0.0.0.0", port=port)

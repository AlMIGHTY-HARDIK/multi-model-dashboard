# minimal_test.py

import plotly.graph_objects as go

# Force removal of "titlefont" from YAxis validators.
validators = go.layout.YAxis.__dict__.get("_validators")
if validators and isinstance(validators, dict):
    go.layout.YAxis._validators = {k: v for k, v in validators.items() if k != "titlefont"}

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the Iris dataset and convert it to a binary classification problem.
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
# Use only classes 0 and 1 to make it binary.
X = X[y < 2]
y = y[y < 2]

# Split the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a RandomForestClassifier.
model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

# Create a ClassifierExplainer with a tree-based SHAP explainer.
explainer = ClassifierExplainer(model, X_test, y_test, shap='tree')

# Build the ExplainerDashboard.
dashboard = ExplainerDashboard(explainer)

# Patch the underlying Dash app to redirect run_server to run.
dashboard.app.run_server = dashboard.app.run

# Launch the dashboard.
dashboard.run(port=8050)

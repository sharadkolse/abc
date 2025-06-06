import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Map algorithm names to sklearn estimators and simple descriptions
ALGORITHMS = {
    "Logistic Regression": "A linear model that estimates probabilities using a logistic function.",
    "Support Vector Machine": "Finds a hyperplane that best separates classes with maximum margin.",
    "Decision Tree": "Partitions data by learning decision rules in a tree structure.",
    "K-Nearest Neighbors": "Classifies a point based on majority vote of its nearest neighbors.",
    "Random Forest": "Ensemble of decision trees aggregated by averaging or voting.",
    "Neural Network": "Feedforward multi-layer perceptron trained with backpropagation.",
}


def create_model(name):
    """Instantiate a model based on sidebar parameters."""
    if name == "Logistic Regression":
        c = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        return LogisticRegression(C=c, max_iter=1000), None
    if name == "Support Vector Machine":
        c = st.sidebar.slider("C", 0.1, 10.0, 1.0)
        gamma = st.sidebar.slider("Gamma", 0.01, 1.0, 0.1)
        return SVC(C=c, gamma=gamma, probability=True), None
    if name == "Decision Tree":
        depth = st.sidebar.slider("Max depth", 1, 20, 5)
        return DecisionTreeClassifier(max_depth=depth), None
    if name == "K-Nearest Neighbors":
        n = st.sidebar.slider("Neighbors", 1, 20, 5)
        return KNeighborsClassifier(n_neighbors=n), None
    if name == "Random Forest":
        trees = st.sidebar.slider("Trees", 10, 200, 100)
        depth = st.sidebar.slider("Max depth", 1, 20, 5)
        return RandomForestClassifier(n_estimators=trees, max_depth=depth), None
    if name == "Neural Network":
        hidden = st.sidebar.slider("Hidden units", 5, 100, 20)
        lr = st.sidebar.slider("Learning rate", 0.0001, 0.1, 0.01)
        epochs = st.sidebar.slider("Epochs", 1, 100, 20)
        mlp = MLPClassifier(hidden_layer_sizes=(hidden,), learning_rate_init=lr, max_iter=1, warm_start=True)
        return mlp, epochs
    return LogisticRegression(), None


def get_dataset():
    """Return features and labels from either generated or drawn data."""
    source = st.sidebar.selectbox("Dataset source", ["Generated Moons", "Draw"], key="source")
    if source == "Generated Moons":
        noise = st.sidebar.slider("Noise", 0.0, 1.0, 0.3)
        X, y = make_moons(noise=noise, random_state=42)
        return X, y

    st.write("Draw class 0 (blue) and class 1 (red)")
    width = 300
    height = 300
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Class 0**")
        canvas0 = st_canvas(
            background_color="white",
            stroke_color="blue",
            drawing_mode="point",
            width=width,
            height=height,
            key="canvas0",
            update_streamlit=True,
        )
    with col2:
        st.markdown("**Class 1**")
        canvas1 = st_canvas(
            background_color="white",
            stroke_color="red",
            drawing_mode="point",
            width=width,
            height=height,
            key="canvas1",
            update_streamlit=True,
        )

    X_list = []
    y_list = []
    if canvas0.json_data is not None:
        for obj in canvas0.json_data.get("objects", []):
            X_list.append([obj["left"], obj["top"]])
            y_list.append(0)
    if canvas1.json_data is not None:
        for obj in canvas1.json_data.get("objects", []):
            X_list.append([obj["left"], obj["top"]])
            y_list.append(1)

    if not X_list:
        return None, None

    X = np.array(X_list, dtype=float)
    y = np.array(y_list)
    X[:, 0] = (X[:, 0] - width / 2) / (width / 2)
    X[:, 1] = (height / 2 - X[:, 1]) / (height / 2)
    return X, y


def plot_decision_boundary(model, X, y):
    """Plot the decision boundary of a classifier along with the data."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary")
    st.pyplot(fig)


def main():
    st.title("ML Algorithms Visualized")
    st.write(
        "Select an algorithm from the sidebar, optionally draw your own dataset, and tweak parameters to see how each model behaves."
    )

    algorithm_name = st.sidebar.selectbox("Algorithm", list(ALGORITHMS.keys()))
    st.sidebar.write(ALGORITHMS[algorithm_name])

    X, y = get_dataset()
    if X is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model, epochs = create_model(algorithm_name)
    pipeline = make_pipeline(StandardScaler(), model)

    if epochs is None:
        pipeline.fit(X_train, y_train)
    else:
        progress = st.progress(0)
        loss_chart = st.line_chart(pd.DataFrame({"loss": []}))
        for i in range(epochs):
            pipeline.fit(X_train, y_train)
            loss = pipeline.named_steps["mlpclassifier"].loss_
            loss_chart.add_rows({"loss": [loss]})
            progress.progress((i + 1) / epochs)
        st.write(f"Final loss: {loss:.4f}")

    st.write(f"### {algorithm_name}")
    plot_decision_boundary(pipeline, X, y)


if __name__ == "__main__":
    main()

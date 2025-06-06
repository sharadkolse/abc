# ML Visualization App

This repository contains a simple Streamlit application that demonstrates how several machine learning algorithms work on a toy dataset.

## Running the app

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Run the Streamlit server:

```bash
streamlit run streamlit_app.py
```

Use the sidebar to select an algorithm and see its decision boundary on a two-dimensional dataset.
You can also draw your own dataset using the built-in canvas component and tweak
algorithm parameters from the sidebar. When the "Neural Network" algorithm is
selected, the app displays the training loss live while it trains.

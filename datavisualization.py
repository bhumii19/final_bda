import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
from sklearn.datasets import load_iris
import os

# Create output folder
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# Random Adult-like dataset
np.random.seed(42)
n = 200
adult = pd.DataFrame({
    "age": np.random.randint(18, 65, n),
    "workclass": np.random.choice(["Private", "Self-Employed", "Government", "Unemployed"], n),
    "education": np.random.choice(["HS-grad", "Bachelors", "Masters", "Doctorate"], n),
    "occupation": np.random.choice(["Tech", "Sales", "Clerical", "Management"], n),
    "hours-per-week": np.random.randint(20, 60, n),
    "income": np.random.choice(["<=50K", ">50K"], n)
})
print("✅ Random Adult-like dataset created successfully!\n")

# Iris dataset
iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris["species"] = iris_data.target_names[iris_data.target]

# 1D Visualization
plt.figure(figsize=(6,4))
sns.histplot(adult["age"], bins=15, kde=True, color="skyblue")
plt.title("1D - Age Distribution (Adult)")
plt.savefig(f"{OUTDIR}/1D_age_distribution.png")
plt.close()

# 2D Visualization
plt.figure(figsize=(6,4))
sns.scatterplot(data=iris, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("2D - Sepal vs Petal Length (Iris)")
plt.savefig(f"{OUTDIR}/2D_scatter.png")
plt.close()

# 3D Visualization
fig = px.scatter_3d(iris, x="sepal length (cm)", y="sepal width (cm)", z="petal length (cm)", color="species",
                    title="3D - Iris Flower Dimensions")
fig.write_html(f"{OUTDIR}/3D_visualization.html")

# Temporal Visualization
temp = adult.groupby("age")["hours-per-week"].mean().reset_index()
plt.figure(figsize=(7,4))
sns.lineplot(data=temp, x="age", y="hours-per-week", color="green")
plt.title("Temporal - Avg Hours/Week by Age (Adult)")
plt.savefig(f"{OUTDIR}/4D_temporal_plot.png")
plt.close()

# Multidimensional Visualization
fig = px.scatter(iris, x="sepal length (cm)", y="petal length (cm)", color="species",
                 size="petal width (cm)", hover_data=["sepal width (cm)"],
                 title="Multidimensional - Iris Dataset")
fig.write_html(f"{OUTDIR}/5D_multidimensional.html")

# Hierarchical Visualization
adult_tree = adult.groupby(["workclass", "education"])["hours-per-week"].mean().reset_index()
fig = px.treemap(adult_tree, path=["workclass", "education"], values="hours-per-week",
                 title="Hierarchical - Workclass → Education (Adult)")
fig.write_html(f"{OUTDIR}/6D_treemap.html")

# Network Visualization
sample = adult[["workclass", "occupation"]].dropna().sample(15, random_state=1)
G = nx.Graph()
for _, row in sample.iterrows():
    G.add_edge(row["workclass"], row["occupation"])
plt.figure(figsize=(6,6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue", edge_color="gray", font_size=9)
plt.title("Network - Workclass vs Occupation (Adult)")
plt.savefig(f"{OUTDIR}/7D_network_graph.png")
plt.close()

print("\n✅ All 7 visualizations saved inside:", OUTDIR)

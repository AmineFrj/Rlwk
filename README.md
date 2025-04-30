# Regularized Lasso-Weighted k-means (RLwk)

**RLWK** is a simple and efficient Python implementation of a feature-weighted K-Means clustering algorithm tailored for text data. It is particularly useful for exploring and interpreting large textual corpora. RLWK introduces a feature weighting mechanism that helps identify the most informative words within each cluster.

This project is lightweight, easy to run, and designed to be used as a base for testing initialization strategies, parameter tuning, or cluster analysis. It also includes tools to assist in visualizing clustering results and supports using Large Language Models (LLMs) like ChatGPT as a post-processing step to summarize and interpret clusters.

## Project Structure

```
RLWK/
├── coclust/               # Implementation of spherical k-means
├── new_data/              # Benchmark datasets
├── RLWK/                  # Main source code
    ├── rlwk.py            # Core functionnalities of Rlwk
    └── top_terms_utils.py # Visualization of top terms utilities
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image
├── docker-compose.yml     # Docker Compose configuration
└── README.md              # This file
```

## Quick Start with Docker

Make sure Docker and Docker Compose are installed on your machine. Then run:

```bash
git clone https://github.com/AmineFrj/Rlwk.git
cd Rlwk
docker-compose up --build
```

This will start a Jupyter Lab server accessible at [http://localhost:8888](http://localhost:8888). The default token and password are : 1234567890

## Manual Execution (Without Docker)

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Lab:

   ```bash
   jupyter lab
   ```

## Key Features

- Implementation of K-Means with various initialization strategies: `random`, `s-kmeans`, `k-means++`
- Performance evaluation using NMI, ARI, and Silhouette Score
- Result visualization and parameter tuning

## RLWK Usage Example

We propose two jupyter notebooks to demonstrate how to use the RLwk algorithm. The
[Simple usage of RLwk](notebooks/Simple%20usage%20of%20RLwk.ipynb) notebook contains simple usage example on benchmark datasets: fit the model, evaluate clustering results, extract top terms, etc. And the
[Visualization of Rlwk results](notebooks/Visualization%20of%20Rlwk%20results.ipynb) notebook contains evaluation experiments.

Here is a basic example of using RLWK with the BBC News dataset, along with an explanation of how to use the core class:

```python
# BBC News Example
from rlwk import RLWK

# Load and transform data
mat_tfidf  # your preprocessed feature matrix
k = 5      # number of clusters

# Initialize and fit the RLWK model
model = RLWK(n_clusters=k, init="s", random_state=42)
model.fit(mat_tfidf)

# Access results
labels = model.labels_            # Cluster labels for each point
weights = model.weights_           # Feature weights
centers = model.cluster_centers_  # Coordinates of cluster centers
inertia = model.P2                # Sum of squared errors
```

**Usage Summary:**

- `fit(X)`: trains the clustering model on your data matrix `X`.
- `.labels_`: provides cluster assignments after fitting.
- `.cluster_centers_`: contains the centroids of the clusters.
- `.inertia_`: evaluates compactness of clustering (lower is better).

You can switch initialization methods with `init="r"` for random, `"s"` fror spherical k-means, or `"k"` for k-means++.python

## 🤝 Contributing

Contributions are welcome! If you have suggestions or bug fixes, feel free to open an issue or a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

```

```

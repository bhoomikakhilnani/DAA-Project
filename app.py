from flask import Flask, request, send_file
import pandas as pd
from io import StringIO, BytesIO
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Global storage
last_graph_data = None
last_source = None
last_distances = None


# ---------- Bellman-Ford ----------
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.edges = []

    def add_edge(self, u, v, w):
        self.edges.append((u, v, w))

    def bellman_ford(self, src):
        dist = {v: float('inf') for v in range(self.V)}
        pred = {v: None for v in range(self.V)}
        dist[src] = 0

        for _ in range(self.V - 1):
            for u, v, w in self.edges:
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u

        for u, v, w in self.edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                raise Exception("Graph contains a negative weight cycle")

        return dist, pred

    @staticmethod
    def from_csv(content):
        df = pd.read_csv(StringIO(content))
        df.columns = df.columns.str.strip().str.lower()

        rename = {}
        for c in df.columns:
            if c in ["from", "start", "src"]:
                rename[c] = "source"
            elif c in ["to", "end", "dest", "destination"]:
                rename[c] = "destination"
            elif c in ["distance", "cost", "weight", "w"]:
                rename[c] = "weight"
        df.rename(columns=rename, inplace=True)

        if not {"source", "destination", "weight"}.issubset(df.columns):
            raise ValueError("CSV must have: source, destination, weight")

        nodes = list(set(df["source"]).union(df["destination"]))
        n2i = {n: i for i, n in enumerate(nodes)}
        i2n = {i: n for n, i in n2i.items()}

        g = Graph(len(nodes))
        for _, r in df.iterrows():
            g.add_edge(n2i[r["source"]], n2i[r["destination"]], float(r["weight"]))
        return g, n2i, i2n


# ---------- Flask ----------
@app.route("/", methods=["GET", "POST"])
def index():
    global last_graph_data, last_source, last_distances

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimal Route Optimizer</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <div class="card shadow p-4">
                <h2 class="text-center mb-4">🚗 Optimal Route Optimizer</h2>
                <form method="POST" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label class="form-label">Upload CSV File</label>
                        <input type="file" name="file" accept=".csv" class="form-control" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Enter Source Vertex (e.g., A or 0)</label>
                        <input type="text" name="source" class="form-control" placeholder="e.g., A" required>
                    </div>
                    <button class="btn btn-primary w-100">Find Optimal Routes</button>
                </form>
                {result}
            </div>
        </div>
    </body>
    </html>
    """

    result = ""
    if request.method == "POST":
        file = request.files.get("file")
        src_name = request.form.get("source").strip()

        try:
            content = file.stream.read().decode("utf-8")
            g, n2i, i2n = Graph.from_csv(content)

            if src_name in n2i:
                src = n2i[src_name]
            elif src_name.isdigit() and int(src_name) in i2n:
                src = int(src_name)
            else:
                raise Exception(f"Source '{src_name}' not found in CSV nodes: {list(n2i.keys())}")

            dist, pred = g.bellman_ford(src)

            last_graph_data = (g, i2n, pred)
            last_source = src
            last_distances = dist

            rows = "".join(
                f"<tr><td>{i2n[v]}</td><td>{'∞' if d == float('inf') else d}</td></tr>"
                for v, d in dist.items()
            )

            result = f"""
            <div class="mt-4">
                <h5>✅ Shortest Distances from Source: {src_name}</h5>
                <table class="table table-striped mt-3">
                    <thead><tr><th>Vertex</th><th>Shortest Distance</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
                <h5 class="mt-4">📊 Graph Visualization</h5>
                <img src="/graph.png" class="img-fluid mt-2 border rounded shadow-sm" alt="Graph Visualization">
            </div>
            """
        except Exception as e:
            result = f'<div class="alert alert-danger mt-3">Error: {e}</div>'

    return html.format(result=result)


@app.route("/graph.png")
def graph_image():
    global last_graph_data, last_source
    if not last_graph_data:
        return "No graph available", 404

    g, i2n, pred = last_graph_data
    src = last_source

    G = nx.DiGraph()
    for u, v, w in g.edges:
        G.add_edge(i2n[u], i2n[v], weight=w)

    pos = nx.spring_layout(G, seed=42)

    shortest_edges = {(i2n[u], i2n[v]) for v, u in pred.items() if u is not None}

    plt.figure(figsize=(6, 4))
    plt.axis("off")

    node_colors = ["red" if n == i2n[src] else "skyblue" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, edgecolors="black")

    # --- Edges with professional arrows ---
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=G.edges,
        edge_color="black",
        width=1.5,
        arrows=True,
        arrowsize=30,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=15,
        min_target_margin=15,
    )

    # --- Highlight shortest path edges ---
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(shortest_edges),
        edge_color="green",
        width=3,
        arrows=True,
        arrowsize=25,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=15,
        min_target_margin=15,
    )

    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", label_pos=0.5)

    plt.title(f"Shortest Paths from Source: {i2n[src]}", fontsize=12)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)






#!/usr/bin/env python3
"""
ADAM's Vector Space Visualizer
See how your 384-dimensional embeddings cluster in 2D space!

This teaches you:
1. How high-dimensional vectors can be visualized
2. Why similar texts cluster together
3. The difference between t-SNE and UMAP
4. How to interpret embedding spaces
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Note: UMAP not installed. Install with: pip install umap-learn")

from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import seaborn as sns

console = Console()

class VectorVisualizer:
    """Visualize high-dimensional embeddings in 2D"""
    
    def __init__(self):
        console.print("[yellow]Loading embedding model for visualization...[/yellow]")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        console.print("[green]✅ Ready to visualize vector spaces![/green]")
        
        # Set visual style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def prepare_sample_texts(self):
        """
        Create diverse text samples to visualize.
        Notice how we group them by type - we expect these to cluster!
        """
        samples = {
            'SQL Queries': [
                "SELECT * FROM users WHERE age > 25",
                "SELECT COUNT(*) FROM orders GROUP BY user_id",
                "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
                "DELETE FROM logs WHERE created_at < '2023-01-01'",
                "UPDATE products SET price = price * 1.1 WHERE category = 'electronics'"
            ],
            'Python Code': [
                "df = pd.read_csv('data.csv')",
                "df.groupby('category').agg({'sales': 'sum'})",
                "for i in range(len(data)): process(data[i])",
                "import numpy as np",
                "plt.plot(x, y, 'ro-')"
            ],
            'Natural Language': [
                "How are you doing today?",
                "The weather is beautiful outside",
                "I love learning about data science",
                "What's your favorite programming language?",
                "Machine learning is fascinating"
            ],
            'Error Messages': [
                "ERROR: relation 'users' does not exist",
                "TypeError: 'NoneType' object is not subscriptable",
                "SyntaxError: unexpected EOF while parsing",
                "ValueError: invalid literal for int()",
                "KeyError: 'user_id'"
            ],
            'Business Questions': [
                "What's our monthly recurring revenue?",
                "How many active users do we have?",
                "Show me the conversion funnel",
                "Calculate customer lifetime value",
                "What's causing user churn?"
            ]
        }
        
        # Flatten into lists
        texts = []
        labels = []
        colors = []
        
        color_map = {
            'SQL Queries': 'blue',
            'Python Code': 'green',
            'Natural Language': 'orange',
            'Error Messages': 'red',
            'Business Questions': 'purple'
        }
        
        for category, items in samples.items():
            texts.extend(items)
            labels.extend([category] * len(items))
            colors.extend([color_map[category]] * len(items))
        
        return texts, labels, colors
    
    def encode_texts(self, texts):
        """
        Convert texts to 384-dimensional vectors.
        This is where meaning becomes geometry!
        """
        console.print(f"\n[cyan]Encoding {len(texts)} texts into 384-dimensional vectors...[/cyan]")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Show some statistics about the embedding space
        console.print(f"\n[green]Embedding Statistics:[/green]")
        console.print(f"  Shape: {embeddings.shape} (texts × dimensions)")
        console.print(f"  Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.3f}")
        console.print(f"  Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.3f}")
        
        return embeddings
    
    def reduce_dimensions_tsne(self, embeddings, perplexity=30):
        """
        t-SNE: Preserves local structure (nearby points stay together)
        
        Mathematical insight: t-SNE minimizes the KL divergence between
        the high-dimensional and low-dimensional probability distributions
        of point neighborhoods.
        """
        # Adjust perplexity to be valid for the number of samples
        n_samples = embeddings.shape[0]
        perplexity = min(perplexity, n_samples - 1)
        
        console.print(f"\n[yellow]Running t-SNE reduction (perplexity={perplexity})...[/yellow]")
        console.print(f"[dim]t-SNE focuses on preserving local neighborhoods[/dim]")
        console.print(f"[dim]Adjusted perplexity to {perplexity} for {n_samples} samples[/dim]")
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        return embeddings_2d
    
    def reduce_dimensions_umap(self, embeddings, n_neighbors=15):
        """
        UMAP: Preserves both local and global structure
        
        Mathematical insight: UMAP constructs a topological representation
        of the high-dimensional data and finds a low-dimensional embedding
        that has the closest possible equivalent topology.
        """
        if not UMAP_AVAILABLE:
            console.print("[red]UMAP not available. Install with: pip install umap-learn[/red]")
            return None
            
        console.print(f"\n[yellow]Running UMAP reduction (n_neighbors={n_neighbors})...[/yellow]")
        console.print("[dim]UMAP preserves both local and global structure[/dim]")
        
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        return embeddings_2d
    
    def visualize_embeddings(self, embeddings_2d, texts, labels, colors, title, method_name):
        """
        Create an interactive visualization of the embedding space
        """
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=colors, alpha=0.7, s=100)
        
        # Add labels for some points to avoid overcrowding
        for i in range(0, len(texts), 3):  # Label every 3rd point
            plt.annotate(texts[i][:30] + "...", 
                        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=8, alpha=0.7)
        
        plt.title(f'{title}\n{method_name} Visualization of 384D → 2D', fontsize=16)
        plt.xlabel(f'{method_name} Component 1')
        plt.ylabel(f'{method_name} Component 2')
        
        # Create legend - FIXED VERSION
        from matplotlib.patches import Patch
        color_map = {
            'SQL Queries': 'blue',
            'Python Code': 'green',
            'Natural Language': 'orange',
            'Error Messages': 'red',
            'Business Questions': 'purple'
        }
        
        # Create legend elements with proper color mapping
        unique_labels = list(dict.fromkeys(labels))  # Preserves order unlike set()
        legend_elements = [Patch(facecolor=color_map[label], label=label) 
                          for label in unique_labels]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        return plt
    
    def analyze_clusters(self, embeddings_2d, labels):
        """
        Analyze how well different text types clustered together
        """
        console.print("\n[bold]Cluster Analysis:[/bold]")
        
        # Calculate centroid for each category
        unique_labels = list(set(labels))
        centroids = {}
        
        for label in unique_labels:
            mask = [l == label for l in labels]
            points = embeddings_2d[mask]
            centroid = np.mean(points, axis=0)
            spread = np.std(points, axis=0)
            
            centroids[label] = centroid
            console.print(f"\n[cyan]{label}:[/cyan]")
            console.print(f"  Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
            console.print(f"  Spread (std): ({spread[0]:.2f}, {spread[1]:.2f})")
        
        # Calculate inter-cluster distances
        console.print("\n[bold]Inter-cluster Distances:[/bold]")
        console.print("(How far apart are different text types?)\n")
        
        for i, label1 in enumerate(unique_labels):
            distances = []
            for j, label2 in enumerate(unique_labels):
                if i != j:
                    dist = np.linalg.norm(centroids[label1] - centroids[label2])
                    distances.append(f"{dist:.2f}")
            console.print(f"{label1:20} → {' '.join(distances)}")
    
    def run_visualization_experiment(self):
        """
        Complete visualization pipeline with educational insights
        """
        console.print(Panel.fit(
            "[bold cyan]Vector Space Visualization Experiment[/bold cyan]\n\n"
            "We'll visualize how different types of text cluster in embedding space.\n"
            "This reveals the geometric structure of meaning!",
            title="🔬 Embedding Visualization"
        ))
        
        # Prepare data
        texts, labels, colors = self.prepare_sample_texts()
        
        # Create embeddings
        embeddings = self.encode_texts(texts)
        
        # t-SNE visualization
        embeddings_tsne = self.reduce_dimensions_tsne(embeddings)
        plt_tsne = self.visualize_embeddings(
            embeddings_tsne, texts, labels, colors,
            "How Different Text Types Cluster in Embedding Space",
            "t-SNE"
        )
        
        # Analyze t-SNE clusters
        console.print("\n[bold magenta]t-SNE Analysis:[/bold magenta]")
        self.analyze_clusters(embeddings_tsne, labels)
        
        # UMAP visualization (if available)
        if UMAP_AVAILABLE:
            embeddings_umap = self.reduce_dimensions_umap(embeddings)
            plt_umap = self.visualize_embeddings(
                embeddings_umap, texts, labels, colors,
                "How Different Text Types Cluster in Embedding Space",
                "UMAP"
            )
            
            console.print("\n[bold magenta]UMAP Analysis:[/bold magenta]")
            self.analyze_clusters(embeddings_umap, labels)
        
        # Show the plots
        plt.show()
        
        # Educational insights
        console.print(Panel.fit(
            "[bold green]What This Visualization Teaches Us:[/bold green]\n\n"
            "1. **Semantic Clustering**: Similar text types naturally cluster together\n"
            "   - SQL queries form one region\n"
            "   - Python code forms another\n"
            "   - This happens automatically from the embeddings!\n\n"
            "2. **Continuous Space**: There are smooth transitions between clusters\n"
            "   - Business questions might be between SQL and natural language\n"
            "   - This reflects their hybrid nature\n\n"
            "3. **Dimensionality Reduction Trade-offs**:\n"
            "   - t-SNE: Better at preserving local structure (tight clusters)\n"
            "   - UMAP: Better at preserving global structure (cluster relationships)\n\n"
            "4. **Why This Matters for ADAM**:\n"
            "   - When you ask a question, ADAM finds the nearest points in this space\n"
            "   - The clustering explains why ADAM can find relevant SQL when you ask in natural language!",
            title="💡 Key Insights"
        ))
    
    def visualize_similarity_gradients(self):
        """
        Show how similarity changes gradually in vector space
        """
        console.print("\n[bold cyan]Similarity Gradient Visualization[/bold cyan]")
        
        # Create a gradient of texts from natural to technical
        gradient_texts = [
            "Hello, how are you?",                          # Very natural
            "What's the user count?",                       # Natural but data-related
            "Show me how many users we have",              # More specific
            "I need to count all users",                    # Getting technical
            "Count the users in the database",             # More technical
            "Get user count from users table",             # Very technical
            "SELECT COUNT(*) FROM users",                   # Pure SQL
        ]
        
        embeddings = self.model.encode(gradient_texts)
        
        # Calculate similarity between adjacent pairs
        console.print("\n[green]Similarity between adjacent texts:[/green]")
        for i in range(len(gradient_texts) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            console.print(f"{i} → {i+1}: {sim:.3f} | {gradient_texts[i][:30]}... → {gradient_texts[i+1][:30]}...")
        
        # Create a heatmap of all similarities
        n = len(gradient_texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=[f"Text {i}" for i in range(n)],
                   yticklabels=[f"Text {i}" for i in range(n)])
        plt.title("Similarity Matrix: Natural Language → SQL Gradient")
        plt.tight_layout()
        plt.show()


def main():
    """Run the visualization experiments"""
    visualizer = VectorVisualizer()
    
    while True:
        console.print("\n[bold]Vector Visualization Options:[/bold]")
        console.print("  1. Visualize text clustering (t-SNE and UMAP)")
        console.print("  2. Explore similarity gradients")
        console.print("  3. Exit")
        
        choice = input("\nChoose an option: ")
        
        if choice == "1":
            visualizer.run_visualization_experiment()
        elif choice == "2":
            visualizer.visualize_similarity_gradients()
        elif choice == "3":
            break
        else:
            console.print("[red]Invalid choice[/red]")


if __name__ == "__main__":
    main()
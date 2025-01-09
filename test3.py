import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def topic_sensitive_pagerank(links, topic_vector, alpha=0.85, max_iter=100, tol=1e-6):
    n = links.shape[0]
    pagerank = np.ones(n) / n
    for _ in range(max_iter):
        new_pagerank = np.zeros(n)
        for i in range(n):
            incoming_links = np.where(links[:, i] > 0)[0] 
            for j in incoming_links:
                new_pagerank[i] += pagerank[j] / np.sum(links[j]) 
            new_pagerank[i] = (1 - alpha) * topic_vector[i] + alpha * new_pagerank[i]  
        if np.linalg.norm(new_pagerank - pagerank, 1) < tol:
            break
        pagerank = new_pagerank
    return pagerank
links = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0]])
topic_vector = np.array([0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
pagerank_scores = topic_sensitive_pagerank(links, topic_vector)
print("Topic-Sensitive PageRank Scores:", pagerank_scores)
G = nx.from_numpy_array(links, create_using=nx.DiGraph)
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=16, font_weight='bold', arrows=True)
plt.title('Đồ thị liên kết')
plt.show()
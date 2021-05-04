Node Embeddings of recipe Graph 
This article introduces node embeddings, their applications and provides steps to get node embeddings of recipe data using components from stellargraph library for Keras implementation of Node2Vec to perform representation learning.
Node embeddings are compressed representation of a graph topology while preserving important network features. Node embeddings represent each node in a graph as a fixed length vector of real numbers in d-dimensions such that similar nodes in the network have embeddings close to each other while capturing key features and reducing dimensionality.  These node embeddings can be used for community detection or clustering, node classifications and link predictions. 
For a given network graph G(V, E), the mapping function f∶V →R^d converts graph G to vector representation of each nodes in d-dimensions. The figure 1 represents the embedding of a node u ∈V  where V represents a set of nodes. 


 
                  Figure 1:  The model architecture of Node2Vec algorithm. 
The node embeddings in new embedding space reflects the optimized relative positions of the nodes in the original graph. For example, 2D embeddings of Zachary’s Karate Club Network are shown in Figure 2. 
 
           Figure 2: Graph of Zachary’s karate club and nodes on embedding space.
In this article, 
Key steps of node embeddings involve choosing:
	Similarity function
	Encoder 
	Decoder (neighborhood reconstruction)
	Loss function 

Similarity function
Intuition of similarity function is to specify how nodes are related in encoded vector space and original network. We encode nodes in such a way that the similarity between nodes in encoding space and the original network is optimized:
Similarity (u,v)  ≈ z_u^(T ) z_v                                                  (1)
Where Similarity (u,v) is the similarity of nodes u and v in the original network, and z_u^(T ) z_v is the dot product between nodes in embedding space.

Encoder
Encoder maps each node (u,v∈V) in low-dimensional space in embedding space in such a way that the unique embedded vectors (z_u,z_v  ∈ R^d) for each node summarizes their corresponding optimized positions the graph.

 
Figure 3: Illustration of how encoder (ENC) maps nodes from original graph network to low-dimensional embedding space.

Decoder
Decoder reconstructs the neighborhood graph network using the node embeddings that are generated using encoder function. 
                 DEC(z_u,z_v )≜  e^(z_u^T z_v )/(∑_(n∈V)▒e^(z_u^T z_n ) )
                                   ≈P_(G,l) (v|u)

Where  P_(G,l) (v|u)  is the probability of visiting v starting from node u on a random walk of length l in graph G. 

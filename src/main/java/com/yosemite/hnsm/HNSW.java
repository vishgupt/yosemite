package com.yosemite.hnsm;

import java.util.*;

/**
 * Hierarchical Navigable Small World (HNSW) implementation for approximate nearest neighbor search.
 * 
 * HNSW is a graph-based algorithm that organizes vectors in a hierarchical structure with multiple layers.
 * Each layer is a navigable small world graph, and the hierarchy enables efficient approximate nearest neighbor search.
 * 
 * Key properties:
 * - Time complexity: O(log N) for search and insert operations
 * - Space complexity: O(N) for storing the graph
 * - Supports approximate nearest neighbor queries
 */
public class HNSW {
    // Map storing all nodes indexed by their vector ID
    private final Map<Integer, Node> nodes;
    
    // Maximum number of connections per node at upper layers
    private final int maxConnections;
    
    // Maximum number of connections per node at layer 0 (bottom layer)
    // Typically 2x maxConnections for better connectivity at the base layer
    private final int maxConnectionsZeroLevel;
    
    // Controls the exponential decay of layer assignment probabilities
    // Typical value: 1.0 / ln(2.0) ≈ 1.443
    private final float levelMultiplier;
    
    // Random number generator for layer assignment
    private final Random random;
    
    // ID of the entry point node (node with highest level)
    private int entryPoint;
    
    // Maximum level currently in the structure
    private int maxLevel;

    /**
     * Constructs an HNSW index with specified parameters.
     * 
     * @param maxConnections maximum connections per node at upper layers
     * @param levelMultiplier controls layer assignment probability (typically 1/ln(2))
     */
    public HNSW(int maxConnections, float levelMultiplier) {
        this.nodes = new HashMap<>();
        this.maxConnections = maxConnections;
        this.maxConnectionsZeroLevel = maxConnections * 2;
        this.levelMultiplier = levelMultiplier;
        this.random = new Random();
        this.entryPoint = -1;
        this.maxLevel = -1;
    }

    /**
     * Inserts a vector into the HNSW structure.
     * 
     * Algorithm:
     * 1. Assign a random level to the new node
     * 2. If this is the first node, set it as entry point
     * 3. Search for nearest neighbors from entry point down to the new node's level
     * 4. At each layer from new node's level down to 0:
     *    - Find candidate neighbors using search
     *    - Connect new node to candidates
     *    - Connect candidates back to new node
     *    - Prune connections if degree exceeds maximum
     * 5. Update entry point if new node has higher level
     * 
     * @param vector the vector to insert
     * @throws IllegalArgumentException if vector with same ID already exists
     */
    public void insert(Vector vector) {
        int vectorId = vector.getId();
        if (nodes.containsKey(vectorId)) {
            throw new IllegalArgumentException("Vector with id " + vectorId + " already exists");
        }

        // Assign random level to new node using exponential decay distribution
        int level = getRandomLevel();
        Node newNode = new Node(vector, level);
        nodes.put(vectorId, newNode);

        // Handle first insertion
        if (entryPoint == -1) {
            entryPoint = vectorId;
            maxLevel = level;
            return;
        }

        // Find nearest neighbor at all levels above the new node's level
        // This helps navigate from entry point down to the new node's level
        int nearest = entryPoint;
        for (int lc = maxLevel; lc > level; lc--) {
            nearest = searchLayer(vector, Collections.singleton(nearest), 1, lc).get(0);
        }

        // Insert at each layer from new node's level down to layer 0
        for (int lc = Math.min(level, maxLevel); lc >= 0; lc--) {
            // Find candidate neighbors at this layer
            List<Integer> candidates = searchLayer(vector, Collections.singleton(nearest), maxConnections, lc);
            
            // Determine maximum connections for this layer
            // Layer 0 has more connections (2x) for better base connectivity
            int m = lc == 0 ? maxConnectionsZeroLevel : maxConnections;

            // Connect new node to candidates and vice versa
            for (int candidate : candidates) {
                newNode.addConnection(lc, candidate);
                Node candidateNode = nodes.get(candidate);
                candidateNode.addConnection(lc, vectorId);

                // Prune candidate's connections if it exceeds maximum degree
                if (candidateNode.getConnectionCount(lc) > m) {
                    pruneConnections(candidate, m, lc);
                }
            }
            
            // Use closest candidate as entry point for next layer down
            nearest = candidates.get(0);
        }

        // Update entry point if new node has higher level
        if (level > maxLevel) {
            maxLevel = level;
            entryPoint = vectorId;
        }
    }

    /**
     * Searches for k nearest neighbors using a SearchRequest.
     * 
     * Algorithm:
     * 1. Start from entry point (highest level node)
     * 2. Traverse down through layers, finding nearest neighbor at each level
     * 3. At layer 0, search for k nearest neighbors
     * 4. Return results sorted by distance
     * 
     * Time complexity: O(log N) on average
     * 
     * @param request the search request containing query vector and parameters
     * @return list of SearchResult objects sorted by distance
     */
    public List<SearchResult> search(SearchRequest request) {
        Vector query = request.getQueryVector();
        int k = request.getTopK();
        
        // Return empty list if index is empty
        if (entryPoint == -1) {
            return new ArrayList<>();
        }

        // Traverse from top layer down to layer 1, finding nearest neighbor at each level
        // This greedy search efficiently navigates the hierarchy
        int nearest = entryPoint;
        for (int lc = maxLevel; lc > 0; lc--) {
            List<Integer> candidates = searchLayer(query, Collections.singleton(nearest), 1, lc);
            if (!candidates.isEmpty()) {
                nearest = candidates.get(0);
            }
        }

        // At layer 0, search for k (or more) nearest neighbors
        // We search for max(k, maxConnections) to ensure we have enough candidates
        List<Integer> result = searchLayer(query, Collections.singleton(nearest), Math.max(k, maxConnections), 0);
        int returnSize = Math.min(k, result.size());
        List<Integer> topK = result.subList(0, returnSize);
        
        // Convert vector IDs to SearchResult objects with distances
        List<SearchResult> searchResults = new ArrayList<>();
        for (int vectorId : topK) {
            float distance = nodes.get(vectorId).getVector().distance(query);
            searchResults.add(new SearchResult(vectorId, distance));
        }
        return searchResults;
    }

    /**
     * Searches for candidates at a specific layer using greedy search.
     * 
     * This is the core search algorithm that explores a navigable small world graph.
     * It maintains a priority queue of candidates and expands the search by exploring
     * neighbors of the closest candidate until no better candidates are found.
     * 
     * Algorithm:
     * 1. Initialize with entry points
     * 2. While candidates exist:
     *    a. Pop closest candidate
     *    b. If candidate is farther than lower bound, stop
     *    c. Explore all unvisited neighbors
     *    d. Add promising neighbors to candidates
     *    e. Maintain top numCandidates results
     * 3. Return results sorted by distance
     * 
     * @param query the query vector
     * @param entryPoints starting points for search
     * @param numCandidates maximum number of candidates to return
     * @param layer the layer to search in
     * @return list of candidate node IDs sorted by distance
     */
    private List<Integer> searchLayer(Vector query, Set<Integer> entryPoints, int numCandidates, int layer) {
        // Track visited nodes to avoid redundant distance calculations
        Set<Integer> visited = new HashSet<>(entryPoints);
        
        // Priority queue of candidates ordered by distance (closest first)
        PriorityQueue<Integer> candidates = new PriorityQueue<>(
            (a, b) -> Float.compare(nodes.get(b).getVector().distance(query),
                                   nodes.get(a).getVector().distance(query))
        );
        
        // Results list maintaining top candidates
        List<Integer> results = new ArrayList<>();

        // Initialize with entry points
        float lowerBound = Float.MAX_VALUE;
        for (int point : entryPoints) {
            float dist = nodes.get(point).getVector().distance(query);
            candidates.add(point);
            results.add(point);
            lowerBound = Math.min(lowerBound, dist);
        }

        // Greedy search: expand from closest candidates
        while (!candidates.isEmpty()) {
            int current = candidates.poll();
            float currentDist = nodes.get(current).getVector().distance(query);
            
            // Stop if current candidate is farther than best result found so far
            if (currentDist > lowerBound) {
                break;
            }

            // Explore all neighbors of current node at this layer
            for (int neighbor : nodes.get(current).getConnections(layer)) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    float dist = nodes.get(neighbor).getVector().distance(query);
                    
                    // Add neighbor if it's promising (better than lower bound or we need more results)
                    if (dist < lowerBound || results.size() < numCandidates) {
                        candidates.add(neighbor);
                        results.add(neighbor);
                        
                        // Maintain top numCandidates results
                        if (results.size() > numCandidates) {
                            // Find and remove the farthest result
                            int farthestIdx = 0;
                            float maxDist = nodes.get(results.get(0)).getVector().distance(query);
                            for (int i = 1; i < results.size(); i++) {
                                float d = nodes.get(results.get(i)).getVector().distance(query);
                                if (d > maxDist) {
                                    maxDist = d;
                                    farthestIdx = i;
                                }
                            }
                            results.remove(farthestIdx);
                            lowerBound = maxDist;
                        } else {
                            lowerBound = Math.min(lowerBound, dist);
                        }
                    }
                }
            }
        }

        // Sort results by distance before returning
        results.sort((a, b) -> Float.compare(nodes.get(a).getVector().distance(query),
                                            nodes.get(b).getVector().distance(query)));
        return results;
    }

    /**
     * Prunes connections to keep only the m closest neighbors.
     * 
     * When a node exceeds its maximum degree, we remove the farthest connections
     * to maintain the small-world property. This is done bidirectionally to keep
     * the graph consistent.
     * 
     * @param nodeId the node to prune
     * @param m maximum number of connections to keep
     * @param layer the layer to prune connections in
     */
    private void pruneConnections(int nodeId, int m, int layer) {
        Node node = nodes.get(nodeId);
        Set<Integer> connections = node.getConnections(layer);

        // No pruning needed if already within limit
        if (connections.size() <= m) {
            return;
        }

        // Sort connections by distance to the node
        Vector nodeVector = node.getVector();
        List<Integer> sorted = new ArrayList<>(connections);
        sorted.sort((a, b) -> Float.compare(nodes.get(a).getVector().distance(nodeVector),
                                            nodes.get(b).getVector().distance(nodeVector)));

        // Keep only the m closest connections
        Set<Integer> newConnections = new HashSet<>(sorted.subList(0, m));
        
        // Collect connections to remove
        List<Integer> toRemove = new ArrayList<>();
        for (int conn : connections) {
            if (!newConnections.contains(conn)) {
                toRemove.add(conn);
            }
        }
        
        // Remove connections bidirectionally
        for (int conn : toRemove) {
            node.removeConnection(layer, conn);
            nodes.get(conn).removeConnection(layer, nodeId);
        }
    }

    /**
     * Generates a random level for a new node using exponential decay distribution.
     * 
     * The level assignment follows an exponential distribution with parameter levelMultiplier.
     * This ensures that most nodes are at low levels (dense), while a few nodes reach high levels.
     * This creates the hierarchical structure essential to HNSW's efficiency.
     * 
     * Formula: level = floor(-ln(uniform_random) * levelMultiplier)
     * 
     * @return random level for a new node
     */
    private int getRandomLevel() {
        return (int) (-Math.log(random.nextDouble()) * levelMultiplier);
    }

    /**
     * Returns the number of vectors in the index.
     * 
     * @return number of vectors
     */
    public int size() {
        return nodes.size();
    }

    /**
     * Checks if a vector with the given ID exists in the index.
     * 
     * @param vectorId the ID to check
     * @return true if vector exists, false otherwise
     */
    public boolean contains(int vectorId) {
        return nodes.containsKey(vectorId);
    }
}

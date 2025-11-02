package com.yosemite.hnsm;

import java.util.*;

/**
 * Hierarchical Navigable Small World (HNSW) implementation for approximate nearest neighbor search.
 */
public class HNSW {
    private final Map<Integer, Node> nodes;
    private final int maxConnections;
    private final int maxConnectionsZeroLevel;
    private final float levelMultiplier;
    private final Random random;
    private int entryPoint;
    private int maxLevel;

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
     */
    public void insert(Vector vector) {
        int vectorId = vector.getId();
        if (nodes.containsKey(vectorId)) {
            throw new IllegalArgumentException("Vector with id " + vectorId + " already exists");
        }

        int level = getRandomLevel();
        Node newNode = new Node(vector, level);
        nodes.put(vectorId, newNode);

        if (entryPoint == -1) {
            entryPoint = vectorId;
            maxLevel = level;
            return;
        }

        // Find nearest neighbors at all levels
        int nearest = entryPoint;
        for (int lc = maxLevel; lc > level; lc--) {
            nearest = searchLayer(vector, Collections.singleton(nearest), 1, lc).get(0);
        }

        for (int lc = Math.min(level, maxLevel); lc >= 0; lc--) {
            List<Integer> candidates = searchLayer(vector, Collections.singleton(nearest), maxConnections, lc);
            int m = lc == 0 ? maxConnectionsZeroLevel : maxConnections;

            for (int candidate : candidates) {
                newNode.addConnection(lc, candidate);
                Node candidateNode = nodes.get(candidate);
                candidateNode.addConnection(lc, vectorId);

                if (candidateNode.getConnectionCount(lc) > m) {
                    pruneConnections(candidate, m, lc);
                }
            }
            nearest = candidates.get(0);
        }

        if (level > maxLevel) {
            maxLevel = level;
            entryPoint = vectorId;
        }
    }

    /**
     * Searches for k nearest neighbors using a SearchRequest.
     */
    public List<SearchResult> search(SearchRequest request) {
        Vector query = request.getQueryVector();
        int k = request.getTopK();
        
        if (entryPoint == -1) {
            return new ArrayList<>();
        }

        int nearest = entryPoint;
        for (int lc = maxLevel; lc > 0; lc--) {
            List<Integer> candidates = searchLayer(query, Collections.singleton(nearest), 1, lc);
            if (!candidates.isEmpty()) {
                nearest = candidates.get(0);
            }
        }

        List<Integer> result = searchLayer(query, Collections.singleton(nearest), Math.max(k, maxConnections), 0);
        int returnSize = Math.min(k, result.size());
        List<Integer> topK = result.subList(0, returnSize);
        
        // Convert to SearchResult objects
        List<SearchResult> searchResults = new ArrayList<>();
        for (int vectorId : topK) {
            float distance = nodes.get(vectorId).getVector().distance(query);
            searchResults.add(new SearchResult(vectorId, distance));
        }
        return searchResults;
    }

    /**
     * Searches for candidates at a specific layer.
     */
    private List<Integer> searchLayer(Vector query, Set<Integer> entryPoints, int numCandidates, int layer) {
        Set<Integer> visited = new HashSet<>(entryPoints);
        PriorityQueue<Integer> candidates = new PriorityQueue<>(
            (a, b) -> Float.compare(nodes.get(b).getVector().distance(query),
                                   nodes.get(a).getVector().distance(query))
        );
        List<Integer> results = new ArrayList<>();

        float lowerBound = Float.MAX_VALUE;
        for (int point : entryPoints) {
            float dist = nodes.get(point).getVector().distance(query);
            candidates.add(point);
            results.add(point);
            lowerBound = Math.min(lowerBound, dist);
        }

        while (!candidates.isEmpty()) {
            int current = candidates.poll();
            float currentDist = nodes.get(current).getVector().distance(query);
            
            if (currentDist > lowerBound) {
                break;
            }

            for (int neighbor : nodes.get(current).getConnections(layer)) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    float dist = nodes.get(neighbor).getVector().distance(query);
                    if (dist < lowerBound || results.size() < numCandidates) {
                        candidates.add(neighbor);
                        results.add(neighbor);
                        
                        if (results.size() > numCandidates) {
                            // Remove the farthest result
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

        results.sort((a, b) -> Float.compare(nodes.get(a).getVector().distance(query),
                                            nodes.get(b).getVector().distance(query)));
        return results;
    }

    /**
     * Prunes connections to keep only the m closest neighbors.
     */
    private void pruneConnections(int nodeId, int m, int layer) {
        Node node = nodes.get(nodeId);
        Set<Integer> connections = node.getConnections(layer);

        if (connections.size() <= m) {
            return;
        }

        Vector nodeVector = node.getVector();
        List<Integer> sorted = new ArrayList<>(connections);
        sorted.sort((a, b) -> Float.compare(nodes.get(a).getVector().distance(nodeVector),
                                            nodes.get(b).getVector().distance(nodeVector)));

        Set<Integer> newConnections = new HashSet<>(sorted.subList(0, m));
        List<Integer> toRemove = new ArrayList<>();
        for (int conn : connections) {
            if (!newConnections.contains(conn)) {
                toRemove.add(conn);
            }
        }
        for (int conn : toRemove) {
            node.removeConnection(layer, conn);
            nodes.get(conn).removeConnection(layer, nodeId);
        }
    }

    /**
     * Generates a random level for a new node.
     */
    private int getRandomLevel() {
        return (int) (-Math.log(random.nextDouble()) * levelMultiplier);
    }

    public int size() {
        return nodes.size();
    }

    public boolean contains(int vectorId) {
        return nodes.containsKey(vectorId);
    }
}

package com.yosemite.hnsm;

import java.util.*;

/**
 * Represents a node in the HNSW graph.
 */
public class Node {
    private final Vector vector;
    private final int level;
    private final Map<Integer, Set<Integer>> connections;

    public Node(Vector vector, int level) {
        this.vector = vector;
        this.level = level;
        this.connections = new HashMap<>();
        for (int i = 0; i <= level; i++) {
            connections.put(i, new HashSet<>());
        }
    }

    public Vector getVector() {
        return vector;
    }

    public int getLevel() {
        return level;
    }

    public Set<Integer> getConnections(int layer) {
        return connections.getOrDefault(layer, new HashSet<>());
    }

    public void addConnection(int layer, int nodeId) {
        connections.get(layer).add(nodeId);
    }

    public void removeConnection(int layer, int nodeId) {
        connections.get(layer).remove(nodeId);
    }

    public boolean hasConnection(int layer, int nodeId) {
        return connections.get(layer).contains(nodeId);
    }

    public int getConnectionCount(int layer) {
        return connections.get(layer).size();
    }
}

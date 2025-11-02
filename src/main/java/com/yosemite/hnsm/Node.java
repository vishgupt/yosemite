package com.yosemite.hnsm;

import java.util.*;

/**
 * Represents a node in the HNSW graph.
 * 
 * Each node contains a vector and maintains connections to other nodes across multiple layers.
 * The hierarchical structure allows efficient navigation through the graph.
 * A node at level L exists in layers 0 through L, with layer 0 being the densest.
 */
public class Node {
    // The vector data stored in this node
    private final Vector vector;
    
    // Maximum layer this node participates in (0 to level)
    private final int level;
    
    // Map from layer number to set of connected node IDs at that layer
    // Each layer maintains a separate set of connections for navigability
    private final Map<Integer, Set<Integer>> connections;

    /**
     * Constructs a Node with the given vector and level.
     * 
     * Initializes empty connection sets for all layers from 0 to level.
     * 
     * @param vector the vector data for this node
     * @param level the maximum layer this node participates in
     */
    public Node(Vector vector, int level) {
        this.vector = vector;
        this.level = level;
        this.connections = new HashMap<>();
        // Initialize connection sets for each layer
        for (int i = 0; i <= level; i++) {
            connections.put(i, new HashSet<>());
        }
    }

    /**
     * Returns the vector stored in this node.
     * 
     * @return the vector
     */
    public Vector getVector() {
        return vector;
    }

    /**
     * Returns the maximum layer this node participates in.
     * 
     * @return the level
     */
    public int getLevel() {
        return level;
    }

    /**
     * Returns the set of node IDs connected to this node at the specified layer.
     * 
     * Returns an empty set if the layer doesn't exist.
     * 
     * @param layer the layer number
     * @return set of connected node IDs
     */
    public Set<Integer> getConnections(int layer) {
        return connections.getOrDefault(layer, new HashSet<>());
    }

    /**
     * Adds a connection to another node at the specified layer.
     * 
     * @param layer the layer number
     * @param nodeId the ID of the node to connect to
     */
    public void addConnection(int layer, int nodeId) {
        connections.get(layer).add(nodeId);
    }

    /**
     * Removes a connection to another node at the specified layer.
     * 
     * @param layer the layer number
     * @param nodeId the ID of the node to disconnect from
     */
    public void removeConnection(int layer, int nodeId) {
        connections.get(layer).remove(nodeId);
    }

    /**
     * Checks if this node is connected to another node at the specified layer.
     * 
     * @param layer the layer number
     * @param nodeId the ID of the node to check
     * @return true if connected, false otherwise
     */
    public boolean hasConnection(int layer, int nodeId) {
        return connections.get(layer).contains(nodeId);
    }

    /**
     * Returns the number of connections at the specified layer.
     * 
     * @param layer the layer number
     * @return number of connections
     */
    public int getConnectionCount(int layer) {
        return connections.get(layer).size();
    }
}

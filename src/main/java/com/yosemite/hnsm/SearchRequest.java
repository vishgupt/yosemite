package com.yosemite.hnsm;

/**
 * Represents a search request with query vector and parameters.
 * 
 * SearchRequest encapsulates all parameters needed to perform a nearest neighbor search.
 * It includes the query vector and configuration parameters for controlling the search behavior.
 * This object is immutable and validated upon construction.
 */
public class SearchRequest {
    // The query vector to search for
    private final Vector queryVector;
    
    // Number of nearest neighbors to return
    private final int topK;
    
    // Maximum depth to search (optional limit on search traversal)
    private final int maxSearchDepth;

    /**
     * Constructs a SearchRequest with query vector and topK parameter.
     * 
     * Uses unlimited search depth by default.
     * 
     * @param queryVector the query vector
     * @param topK number of nearest neighbors to return
     * @throws IllegalArgumentException if topK <= 0
     */
    public SearchRequest(Vector queryVector, int topK) {
        this(queryVector, topK, Integer.MAX_VALUE);
    }

    /**
     * Constructs a SearchRequest with all parameters.
     * 
     * @param queryVector the query vector
     * @param topK number of nearest neighbors to return
     * @param maxSearchDepth maximum search depth limit
     * @throws IllegalArgumentException if topK <= 0 or maxSearchDepth <= 0
     */
    public SearchRequest(Vector queryVector, int topK, int maxSearchDepth) {
        if (topK <= 0) {
            throw new IllegalArgumentException("topK must be greater than 0");
        }
        if (maxSearchDepth <= 0) {
            throw new IllegalArgumentException("maxSearchDepth must be greater than 0");
        }
        this.queryVector = queryVector;
        this.topK = topK;
        this.maxSearchDepth = maxSearchDepth;
    }

    /**
     * Returns the query vector.
     * 
     * @return the query vector
     */
    public Vector getQueryVector() {
        return queryVector;
    }

    /**
     * Returns the number of nearest neighbors to return.
     * 
     * @return topK value
     */
    public int getTopK() {
        return topK;
    }

    /**
     * Returns the maximum search depth limit.
     * 
     * @return maxSearchDepth value
     */
    public int getMaxSearchDepth() {
        return maxSearchDepth;
    }

    /**
     * Returns a string representation of this search request.
     * 
     * @return string in format "SearchRequest{queryVector=X, topK=Y, maxSearchDepth=Z}"
     */
    @Override
    public String toString() {
        return "SearchRequest{" +
                "queryVector=" + queryVector.getId() +
                ", topK=" + topK +
                ", maxSearchDepth=" + maxSearchDepth +
                '}';
    }
}

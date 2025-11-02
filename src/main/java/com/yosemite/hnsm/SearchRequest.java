package com.yosemite.hnsm;

/**
 * Represents a search request with query vector and parameters.
 */
public class SearchRequest {
    private final Vector queryVector;
    private final int topK;
    private final int maxSearchDepth;

    public SearchRequest(Vector queryVector, int topK) {
        this(queryVector, topK, Integer.MAX_VALUE);
    }

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

    public Vector getQueryVector() {
        return queryVector;
    }

    public int getTopK() {
        return topK;
    }

    public int getMaxSearchDepth() {
        return maxSearchDepth;
    }

    @Override
    public String toString() {
        return "SearchRequest{" +
                "queryVector=" + queryVector.getId() +
                ", topK=" + topK +
                ", maxSearchDepth=" + maxSearchDepth +
                '}';
    }
}

package com.yosemite.hnsm;

/**
 * Represents a search result with vector ID and distance.
 */
public class SearchResult implements Comparable<SearchResult> {
    private final int vectorId;
    private final float distance;

    public SearchResult(int vectorId, float distance) {
        this.vectorId = vectorId;
        this.distance = distance;
    }

    public int getVectorId() {
        return vectorId;
    }

    public float getDistance() {
        return distance;
    }

    @Override
    public int compareTo(SearchResult other) {
        return Float.compare(this.distance, other.distance);
    }

    @Override
    public String toString() {
        return "SearchResult{" +
                "vectorId=" + vectorId +
                ", distance=" + distance +
                '}';
    }
}

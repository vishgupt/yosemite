package com.yosemite.hnsm;

/**
 * Represents a search result with vector ID and distance.
 * 
 * SearchResult encapsulates the outcome of a nearest neighbor search,
 * containing the ID of the found vector and its distance from the query.
 * Results are comparable and sortable by distance.
 */
public class SearchResult implements Comparable<SearchResult> {
    // The ID of the vector found in the search
    private final int vectorId;
    
    // The distance from the query vector to this result
    private final float distance;

    /**
     * Constructs a SearchResult with the given vector ID and distance.
     * 
     * @param vectorId the ID of the found vector
     * @param distance the distance from the query
     */
    public SearchResult(int vectorId, float distance) {
        this.vectorId = vectorId;
        this.distance = distance;
    }

    /**
     * Returns the ID of the found vector.
     * 
     * @return vector ID
     */
    public int getVectorId() {
        return vectorId;
    }

    /**
     * Returns the distance from the query vector.
     * 
     * @return distance value
     */
    public float getDistance() {
        return distance;
    }

    /**
     * Compares this result with another by distance.
     * 
     * Results are ordered by distance in ascending order (closest first).
     * 
     * @param other the other SearchResult to compare
     * @return negative if this is closer, positive if farther, 0 if equal distance
     */
    @Override
    public int compareTo(SearchResult other) {
        return Float.compare(this.distance, other.distance);
    }

    /**
     * Returns a string representation of this search result.
     * 
     * @return string in format "SearchResult{vectorId=X, distance=Y}"
     */
    @Override
    public String toString() {
        return "SearchResult{" +
                "vectorId=" + vectorId +
                ", distance=" + distance +
                '}';
    }
}

package com.yosemite.hnsm;

import java.util.Arrays;

/**
 * Represents a vector in the HNSW space.
 * 
 * A Vector is an immutable data structure that holds a unique identifier and
 * a fixed-size array of float values representing coordinates in a high-dimensional space.
 * Vectors support distance and similarity calculations used by the HNSW algorithm.
 */
public class Vector {
    // Unique identifier for this vector
    private final float[] data;
    
    // Vector coordinates in high-dimensional space
    private final int id;

    /**
     * Constructs a Vector with the given ID and data.
     * 
     * The data array is copied to ensure immutability.
     * 
     * @param id unique identifier for this vector
     * @param data array of float coordinates (will be copied)
     */
    public Vector(int id, float[] data) {
        this.id = id;
        // Copy data to prevent external modification
        this.data = Arrays.copyOf(data, data.length);
    }

    /**
     * Returns the unique identifier of this vector.
     * 
     * @return vector ID
     */
    public int getId() {
        return id;
    }

    /**
     * Returns a copy of the vector data.
     * 
     * @return array of float coordinates
     */
    public float[] getData() {
        return data;
    }

    /**
     * Returns the dimensionality of this vector.
     * 
     * @return number of dimensions
     */
    public int getDimension() {
        return data.length;
    }

    /**
     * Computes the Euclidean distance to another vector.
     * 
     * Formula: sqrt(sum((a[i] - b[i])^2))
     * 
     * @param other the other vector
     * @return Euclidean distance
     * @throws IllegalArgumentException if vectors have different dimensions
     */
    public float distance(Vector other) {
        if (data.length != other.data.length) {
            throw new IllegalArgumentException("Vectors must have the same dimension");
        }
        float sum = 0;
        for (int i = 0; i < data.length; i++) {
            float diff = data[i] - other.data[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    /**
     * Computes the cosine similarity to another vector.
     * 
     * Formula: (a · b) / (||a|| * ||b||)
     * Returns 0 if either vector has zero norm.
     * Range: [-1, 1] where 1 means identical direction, -1 means opposite.
     * 
     * @param other the other vector
     * @return cosine similarity
     * @throws IllegalArgumentException if vectors have different dimensions
     */
    public float cosineSimilarity(Vector other) {
        if (data.length != other.data.length) {
            throw new IllegalArgumentException("Vectors must have the same dimension");
        }
        float dotProduct = 0;
        float normA = 0;
        float normB = 0;
        for (int i = 0; i < data.length; i++) {
            dotProduct += data[i] * other.data[i];
            normA += data[i] * data[i];
            normB += other.data[i] * other.data[i];
        }
        normA = (float) Math.sqrt(normA);
        normB = (float) Math.sqrt(normB);
        if (normA == 0 || normB == 0) {
            return 0;
        }
        return dotProduct / (normA * normB);
    }
}

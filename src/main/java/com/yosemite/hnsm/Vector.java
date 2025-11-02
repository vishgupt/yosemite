package com.yosemite.hnsm;

import java.util.Arrays;

/**
 * Represents a vector in the HNSW space.
 */
public class Vector {
    private final float[] data;
    private final int id;

    public Vector(int id, float[] data) {
        this.id = id;
        this.data = Arrays.copyOf(data, data.length);
    }

    public int getId() {
        return id;
    }

    public float[] getData() {
        return data;
    }

    public int getDimension() {
        return data.length;
    }

    /**
     * Computes Euclidean distance to another vector.
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
     * Computes cosine similarity to another vector.
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

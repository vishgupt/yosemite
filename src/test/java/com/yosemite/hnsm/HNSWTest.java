package com.yosemite.hnsm;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.List;
import java.util.stream.Collectors;

public class HNSWTest {

    @Test
    public void testInsertAndSearch() {
        HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));

        // Create and insert vectors
        Vector v1 = new Vector(1, new float[]{0.0f, 0.0f});
        Vector v2 = new Vector(2, new float[]{1.0f, 0.0f});
        Vector v3 = new Vector(3, new float[]{0.0f, 1.0f});
        Vector v4 = new Vector(4, new float[]{1.0f, 1.0f});

        hnsw.insert(v1);
        hnsw.insert(v2);
        hnsw.insert(v3);
        hnsw.insert(v4);

        assertEquals(4, hnsw.size());

        // Search for nearest neighbors to v1
        Vector query = new Vector(0, new float[]{0.1f, 0.1f});
        SearchRequest request = new SearchRequest(query, 2);
        List<SearchResult> results = hnsw.search(request);

        assertFalse(results.isEmpty());
        assertTrue(results.size() <= 2);
    }

    @Test
    public void testVectorDistance() {
        Vector v1 = new Vector(1, new float[]{0.0f, 0.0f});
        Vector v2 = new Vector(2, new float[]{3.0f, 4.0f});

        float distance = v1.distance(v2);
        assertEquals(5.0f, distance, 0.001f);
    }

    @Test
    public void testVectorCosineSimilarity() {
        Vector v1 = new Vector(1, new float[]{1.0f, 0.0f});
        Vector v2 = new Vector(2, new float[]{1.0f, 0.0f});

        float similarity = v1.cosineSimilarity(v2);
        assertEquals(1.0f, similarity, 0.001f);
    }

    @Test
    public void testDuplicateInsert() {
        HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));
        Vector v1 = new Vector(1, new float[]{0.0f, 0.0f});

        hnsw.insert(v1);
        assertThrows(IllegalArgumentException.class, () -> hnsw.insert(v1));
    }

    @Test
    public void testSearchEmpty() {
        HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));
        Vector query = new Vector(0, new float[]{0.0f, 0.0f});

        SearchRequest request = new SearchRequest(query, 5);
        List<SearchResult> results = hnsw.search(request);
        assertTrue(results.isEmpty());
    }

    @Test
    public void testLargeDataset() {
        HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));

        // Insert 100 random vectors
        for (int i = 0; i < 100; i++) {
            float[] data = new float[10];
            for (int j = 0; j < 10; j++) {
                data[j] = (float) Math.random();
            }
            hnsw.insert(new Vector(i, data));
        }

        assertEquals(100, hnsw.size());

        // Search for nearest neighbors
        float[] queryData = new float[10];
        for (int i = 0; i < 10; i++) {
            queryData[i] = (float) Math.random();
        }
        Vector query = new Vector(-1, queryData);
        SearchRequest request = new SearchRequest(query, 10);
        List<SearchResult> results = hnsw.search(request);

        assertTrue(results.size() <= 10);
        assertTrue(results.size() > 0);
    }

    @Test
    public void testSearchWithRequest() {
        HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));

        // Create and insert vectors
        Vector v1 = new Vector(1, new float[]{0.0f, 0.0f});
        Vector v2 = new Vector(2, new float[]{1.0f, 0.0f});
        Vector v3 = new Vector(3, new float[]{0.0f, 1.0f});
        Vector v4 = new Vector(4, new float[]{1.0f, 1.0f});

        hnsw.insert(v1);
        hnsw.insert(v2);
        hnsw.insert(v3);
        hnsw.insert(v4);

        // Create a search request
        Vector query = new Vector(0, new float[]{0.1f, 0.1f});
        SearchRequest request = new SearchRequest(query, 2);

        // Search using SearchRequest
        List<SearchResult> results = hnsw.search(request);

        assertFalse(results.isEmpty());
        assertTrue(results.size() <= 2);
        
        // Verify results are sorted by distance
        for (int i = 0; i < results.size() - 1; i++) {
            assertTrue(results.get(i).getDistance() <= results.get(i + 1).getDistance());
        }
    }

    @Test
    public void testSearchRequestValidation() {
        Vector query = new Vector(0, new float[]{0.0f, 0.0f});
        
        assertThrows(IllegalArgumentException.class, () -> new SearchRequest(query, 0));
        assertThrows(IllegalArgumentException.class, () -> new SearchRequest(query, -1));
        assertThrows(IllegalArgumentException.class, () -> new SearchRequest(query, 5, 0));
    }

    @Test
    public void testVectorDimension() {
        Vector v = new Vector(1, new float[]{1.0f, 2.0f, 3.0f});
        assertEquals(3, v.getDimension());
    }

    @Test
    public void testVectorId() {
        Vector v = new Vector(42, new float[]{1.0f, 2.0f});
        assertEquals(42, v.getId());
    }

    @Test
    public void testVectorDistanceMismatchedDimensions() {
        Vector v1 = new Vector(1, new float[]{1.0f, 2.0f});
        Vector v2 = new Vector(2, new float[]{1.0f, 2.0f, 3.0f});
        
        assertThrows(IllegalArgumentException.class, () -> v1.distance(v2));
    }

    @Test
    public void testVectorCosineSimilarityMismatchedDimensions() {
        Vector v1 = new Vector(1, new float[]{1.0f, 2.0f});
        Vector v2 = new Vector(2, new float[]{1.0f, 2.0f, 3.0f});
        
        assertThrows(IllegalArgumentException.class, () -> v1.cosineSimilarity(v2));
    }

    @Test
    public void testVectorCosineSimilarityZeroVector() {
        Vector v1 = new Vector(1, new float[]{0.0f, 0.0f});
        Vector v2 = new Vector(2, new float[]{1.0f, 1.0f});
        
        float similarity = v1.cosineSimilarity(v2);
        assertEquals(0.0f, similarity, 0.001f);
    }

    @Test
    public void testVectorCosineSimilarityOrthogonal() {
        Vector v1 = new Vector(1, new float[]{1.0f, 0.0f});
        Vector v2 = new Vector(2, new float[]{0.0f, 1.0f});
        
        float similarity = v1.cosineSimilarity(v2);
        assertEquals(0.0f, similarity, 0.001f);
    }

    @Test
    public void testVectorCosineSimilarityOpposite() {
        Vector v1 = new Vector(1, new float[]{1.0f, 0.0f});
        Vector v2 = new Vector(2, new float[]{-1.0f, 0.0f});
        
        float similarity = v1.cosineSimilarity(v2);
        assertEquals(-1.0f, similarity, 0.001f);
    }

    @Test
    public void testNodeConnections() {
        Vector v = new Vector(1, new float[]{0.0f, 0.0f});
        Node node = new Node(v, 2);
        
        assertEquals(2, node.getLevel());
        assertEquals(v, node.getVector());
        
        // Test connections at different layers
        node.addConnection(0, 2);
        node.addConnection(0, 3);
        node.addConnection(1, 4);
        
        assertTrue(node.hasConnection(0, 2));
        assertTrue(node.hasConnection(0, 3));
        assertTrue(node.hasConnection(1, 4));
        assertFalse(node.hasConnection(1, 2));
        
        assertEquals(2, node.getConnectionCount(0));
        assertEquals(1, node.getConnectionCount(1));
        
        node.removeConnection(0, 2);
        assertFalse(node.hasConnection(0, 2));
        assertEquals(1, node.getConnectionCount(0));
    }

    @Test
    public void testSearchResultComparable() {
        SearchResult r1 = new SearchResult(1, 1.5f);
        SearchResult r2 = new SearchResult(2, 2.5f);
        SearchResult r3 = new SearchResult(3, 1.5f);
        
        assertTrue(r1.compareTo(r2) < 0);
        assertTrue(r2.compareTo(r1) > 0);
        assertEquals(0, r1.compareTo(r3));
    }

    @Test
    public void testSearchResultToString() {
        SearchResult result = new SearchResult(5, 3.14f);
        String str = result.toString();
        
        assertTrue(str.contains("5"));
        assertTrue(str.contains("3.14"));
    }

    @Test
    public void testSearchRequestToString() {
        Vector query = new Vector(1, new float[]{0.0f, 0.0f});
        SearchRequest request = new SearchRequest(query, 5, 100);
        String str = request.toString();
        
        assertTrue(str.contains("topK=5"));
        assertTrue(str.contains("maxSearchDepth=100"));
    }

    @Test
    public void testSearchRequestDefaultMaxSearchDepth() {
        Vector query = new Vector(1, new float[]{0.0f, 0.0f});
        SearchRequest request = new SearchRequest(query, 5);
        
        assertEquals(Integer.MAX_VALUE, request.getMaxSearchDepth());
    }

    @Test
    public void testHNSWContains() {
        HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));
        Vector v1 = new Vector(1, new float[]{0.0f, 0.0f});
        
        assertFalse(hnsw.contains(1));
        hnsw.insert(v1);
        assertTrue(hnsw.contains(1));
        assertFalse(hnsw.contains(2));
    }

    @Test
    public void testSingleVectorSearch() {
        HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));
        Vector v1 = new Vector(1, new float[]{0.0f, 0.0f});
        hnsw.insert(v1);
        
        Vector query = new Vector(0, new float[]{0.1f, 0.1f});
        SearchRequest request = new SearchRequest(query, 1);
        List<SearchResult> results = hnsw.search(request);
        
        assertEquals(1, results.size());
        assertEquals(1, results.get(0).getVectorId());
    }

    @Test
    public void testSearchMoreResultsThanAvailable() {
        HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));
        Vector v1 = new Vector(1, new float[]{0.0f, 0.0f});
        Vector v2 = new Vector(2, new float[]{1.0f, 0.0f});
        hnsw.insert(v1);
        hnsw.insert(v2);
        
        Vector query = new Vector(0, new float[]{0.05f, 0.05f});
        SearchRequest request = new SearchRequest(query, 10);
        List<SearchResult> results = hnsw.search(request);
        
        assertEquals(2, results.size());
    }

    @Test
    public void testMultipleInsertionsWithDifferentLevels() {
        HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));
        
        // Insert multiple vectors to test level assignment
        for (int i = 0; i < 20; i++) {
            float[] data = new float[5];
            for (int j = 0; j < 5; j++) {
                data[j] = (float) Math.random();
            }
            hnsw.insert(new Vector(i, data));
        }
        
        assertEquals(20, hnsw.size());
        
        // Verify search works with multiple levels
        float[] queryData = new float[5];
        for (int i = 0; i < 5; i++) {
            queryData[i] = (float) Math.random();
        }
        Vector query = new Vector(-1, queryData);
        SearchRequest request = new SearchRequest(query, 5);
        List<SearchResult> results = hnsw.search(request);
        
        assertTrue(results.size() > 0);
        assertTrue(results.size() <= 5);
    }

    @Test
    public void testSearchResultsAreDistinct() {
        HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));
        
        for (int i = 0; i < 10; i++) {
            float[] data = new float[3];
            for (int j = 0; j < 3; j++) {
                data[j] = (float) Math.random();
            }
            hnsw.insert(new Vector(i, data));
        }
        
        Vector query = new Vector(-1, new float[]{0.5f, 0.5f, 0.5f});
        SearchRequest request = new SearchRequest(query, 5);
        List<SearchResult> results = hnsw.search(request);
        
        // Check all results are distinct
        for (int i = 0; i < results.size(); i++) {
            for (int j = i + 1; j < results.size(); j++) {
                assertNotEquals(results.get(i).getVectorId(), results.get(j).getVectorId());
            }
        }
    }

    @Test
    public void testVectorDataCopy() {
        float[] data = new float[]{1.0f, 2.0f, 3.0f};
        Vector v = new Vector(1, data);
        
        // Modify original data
        data[0] = 999.0f;
        
        // Vector should have original data
        assertEquals(1.0f, v.getData()[0], 0.001f);
    }
}

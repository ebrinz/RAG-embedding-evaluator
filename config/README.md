# Notes:

For statistically significant evaluation, we should have a balanced and substantial set of test cases. 

Given n=1000 content samples...

#### Number of Test Queries:


At least 100 queries (10% of content size) to get reliable metrics
Each query should have multiple relevant documents
Should cover different similarity patterns (highly similar, somewhat similar, not similar)


#### Distribution Suggestion:


10-20% of documents should be relevant to each query (on average)
This means about 100-200 documents potentially relevant to each query type
This allows testing both precision and recall meaningfully



## Test case template: test_cases.json

```
{
  "version": "1.0",
  "description": "Test cases for embedding model evaluation",
  "test_requirements": {
    "min_queries_per_category": 20,
    "max_query_content_ratio": 0.1,
    "min_relevant_per_query": 5
  },
  "similarity_distribution": {
    "high": 0.2,
    "medium": 0.3,
    "low": 0.5
  },
  "cases": [
    {
      "name": "exact_match",
      "description": "Testing exact semantic matches",
      "query": "Sample query with exact match",
      "content": "Sample content that exactly matches the query context",
      "expected_relevance": 1.0,
      "metadata": {
        "category": "exact",
        "priority": "high"
      }
    },
    {
      "name": "partial_match",
      "description": "Testing partial semantic overlap",
      "query": "Query about specific topic",
      "content": "Content that partially covers the topic with some variations",
      "expected_relevance": 0.7,
      "metadata": {
        "category": "partial",
        "priority": "medium"
      }
    },
    {
      "name": "metadata_match",
      "description": "Testing metadata-based matching",
      "query": "Query with specific metadata",
      "content": "Content with matching metadata but different context",
      "expected_relevance": 0.5,
      "metadata": {
        "category": "metadata",
        "priority": "medium"
      }
    },
    {
      "name": "no_match",
      "description": "Testing unrelated content",
      "query": "Query about unrelated topic",
      "content": "Content about completely different subject",
      "expected_relevance": 0.0,
      "metadata": {
        "category": "negative",
        "priority": "low"
      }
    }
  ]
}
```

#### Statistical Requirements:

```
"test_requirements": {
    "min_queries_per_category": 20,     // Minimum test queries per category (e.g., per genre)
    "max_query_content_ratio": 0.1,     // Don't let queries be more than 10% of content
    "min_relevant_per_query": 5         // Each query should have at least 5 relevant matches
}
```

#### Similarity Distribution:

```
"similarity_distribution": {
    "high": 0.2,    // 20% of pairs should be highly similar
    "medium": 0.3,  // 30% moderately similar
    "low": 0.5      // 50% low similarity
}
```

#### Test Cases - Here's how to write them for different scenarios:

```
"cases": [
    {
        "name": "same_genre_similar_plot",
        "description": "Testing movies with same genre and similar plot points",
        "query": "Find sci-fi movies about time travel and paradoxes",
        "content": "A brilliant physicist builds a time machine and travels to the future, only to discover his actions have created a temporal paradox threatening reality itself.",
        "expected_relevance": 1.0,
        "metadata": {
            "category": "genre_and_plot",
            "priority": "high"
        }
    },
    {
        "name": "same_genre_different_plot",
        "description": "Testing movies with same genre but different plots",
        "query": "Find sci-fi movies about alien invasion",
        "content": "A team of astronauts discovers an ancient artifact on Mars that holds the key to humanity's origins.",
        "expected_relevance": 0.7,
        "metadata": {
            "category": "genre_only",
            "priority": "medium"
        }
    },
    {
        "name": "different_genre_similar_theme",
        "description": "Testing cross-genre movies with similar themes",
        "query": "Find movies about redemption and second chances",
        "content": "A former criminal tries to rebuild his life and reconnect with his estranged family after being released from prison.",
        "expected_relevance": 0.5,
        "metadata": {
            "category": "thematic",
            "priority": "medium"
        }
    }
]
```

#### Tips for creating good test cases:

- Cover edge cases
- Include diverse content lengths
- Test different similarity types:
   - Exact matches
   - Thematic similarity
   - Structural similarity
   - Metadata matches
- Include negative examples (completely unrelated pairs)
- Consider your domain-specific needs

#### Example for a movie recommendation system:
```
{
    "name": "cross_genre_franchise",
    "description": "Testing franchise recognition across genres",
    "query": "Find movies like 'Star Wars: A New Hope' (Sci-Fi, Adventure)",
    "content": "A young warrior trains in ancient martial arts while uncovering a plot that threatens her peaceful kingdom.",
    "expected_relevance": 0.6,
    "metadata": {
        "category": "franchise_patterns",
        "priority": "medium",
        "notes": "Tests hero's journey pattern recognition"
    }
}
```

#### Remember to:

- Make test cases representative of real queries
- Include both obvious and subtle matches
- Document your reasoning in descriptions
- Use meaningful names for test cases
- Group related tests with metadata categories



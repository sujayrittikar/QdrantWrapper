# This is an example usage of the QdrantWrapper class:

from qdrantWrapper import *

cluster_url = "http://localhost:6333"
api_key = ""
collection_name = "test_collection"
distance = "dot"


qdrant_wrapper_obj = QdrantWrapper(cluster_url, api_key, distance=distance, vector_size=4, print_logs=True)

# If the collection name is to be initialized, just pass it to the constructor
# qdrant_wrapper = QdrantWrapper(cluster_url, api_key, distance=distance, vector_size=4, collection_name=collection_name)

# Create a collection
qdrant_wrapper_obj.create_collection(collection_name)

pointStructs = [
    {
        "id": 1,
        "vector": [0.1, 0.2, 0.3, 0.4]
    },
    {
        "id": 2,
        "vector": [0.2, 0.3, 0.4, 0.5]
    }
]

# Upsert data
qdrant_wrapper_obj.upsert_data(pointStructs)

# Upsert data in batches
qdrant_wrapper_obj.upsert_data_batches(pointStructs, batch_size=1000)

# Search
qdrant_wrapper_obj.search(
    query_vector=[0.1, 0.2, 0.3, 0.4],
    top_k=10
)

# Search with filters
qdrant_wrapper_obj.search(
    query_vector=[0.1, 0.2, 0.3, 0.4],
    top_k=10,
    filters=Filter(
        must=[FieldCondition(
            field="id",
            range=Range(
                gte=1,
                lte=2
            )
        )]
    )
)

# Delete data using ids
qdrant_wrapper_obj.delete_data([1, 2])

# Delete Collection
qdrant_wrapper_obj.delete_collection()


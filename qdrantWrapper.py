'''
    This module provides a wrapper around the Qdrant client to make it easier to use.
    It provides methods to create a collection, upsert data, upsert data in batches and search for similar vectors.
'''

from typing import List, Optional, Sequence, Tuple, Union
from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrantWrapperExceptions import CollectionNameNotProvidedException, CollectionAlreadyExistsException
from qdrant_client.http.models import Filter, FieldCondition, Range


class QdrantWrapper:
    def __init__(
            self,
            cluster_url: str,
            api_key: str,
            distance = Distance.DOT,
            vector_size: int = 1536,
            collection_name: str = None,
            print_logs: bool = False
        ) -> None:
        '''
            Args:
                cluster_url (str): Qdrant cluster url
                api_key (str): Qdrant api key
                collection_name (str): Name of the collection
                distance (Distance, optional): Distance metric. Defaults to Distance.DOT.
                vector_size (int, optional): Size of the vector. Defaults to 1536.
        '''
        self.distance = distance
        self.vector_size = vector_size
        self.qdrant_client = QdrantClient(url=cluster_url, api_key=api_key)
        self.collection_name = collection_name
        self.print_logs = print_logs


    def create_collection(self, collection_name: str, set_collection_name=True) -> None:
        '''
            Create a collection in Qdrant

            Args:
                collection_name (str): Name of the collection
                set_collection_name (bool, optional): Whether to set the collection name of the wrapper object. Defaults to False.
        '''
        if not collection_name:
            raise CollectionNameNotProvidedException("Collection name not provided")

        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
            )
        except UnexpectedResponse:
            raise CollectionAlreadyExistsException("Collection already exists in the DB.")
        if set_collection_name:
            self.collection_name = collection_name
        if self.print_logs:
            print(f"Collection {collection_name} created successfully")

    
    def upsert_data(
            self,
            pointStructs: types.Points,
            collection_name: str = None
        ) -> None:
        '''
            Upsert data into the collection

            Args:
                pointStructs (types.Points): List of points to upsert
                collection_name (str, optional): Name of the collection. Defaults to None.
            
            Raises:
                CollectionNameNotProvidedException: Collection name not provided
        '''
        if collection_name:
            operation_info = self.qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=pointStructs
            )
        elif self.collection_name:
            operation_info = self.qdrant_client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=pointStructs
            )
        else:
            raise CollectionNameNotProvidedException("Collection name not provided")

        if self.print_logs:
            print(operation_info)
            print(f"Upserted {len(pointStructs)} points successfully")


    def upsert_data_batches(
            self,
            pointStructs: types.Points,
            batch_size: int = 50,
            collection_name: str = None
    ) -> None:
        '''
            Upsert data into the collection in batches

            Args:
                pointStructs (types.Points): List of points to upsert
                batch_size (int, optional): Size of the batch. Defaults to 50.
                collection_name (str, optional): Name of the collection. Defaults to None.

            Raises:
                CollectionNameNotProvidedException: Collection name not provided
        '''
        if collection_name:
            for i in range(0, len(pointStructs), batch_size):
                points_batch = pointStructs[i:i+batch_size]
                self.upsert_data(points_batch, collection_name)
        elif self.collection_name:
            for i in range(0, len(pointStructs), batch_size):
                points_batch = pointStructs[i:i+batch_size]
                self.upsert_data(points_batch, self.collection_name)
        else:
            raise CollectionNameNotProvidedException("Collection name not provided")


    def search(
            self,
            query_vector: Union[
                types.NumpyArray,
                Sequence[float],
                Tuple[str, List[float]],
                types.NamedVector,
                types.NamedSparseVector,
            ],
            limit=10,
            query_filter: Optional[types.Filter] = None,
            collection_name: str = None
        ) -> list:
        '''
            Search for similar vectors in the collection

            Args:
                query_vector (Union[types.NumpyArray, Sequence[float], Tuple[str, List[float]], types.NamedVector, types.NamedSparseVector]): Query vector
                limit (int, optional): Number of results to return. Defaults to 3.
                query_filter (Optional[types.Filter], optional): Query filter. Defaults to None.
                collection_name (str, optional): Name of the collection. Defaults to None.

            Returns:
                list: List of search results
        '''
        if collection_name:
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter
            )
        elif self.collection_name:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
        else:
            raise CollectionNameNotProvidedException("Collection name not provided")
        return search_result


    def delete_collection(self, collection_name: str = None) -> None:
        '''
            Delete a collection from Qdrant

            Args:
                collection_name (str, optional): Name of the collection. Defaults to None.
        '''
        if collection_name:
            self.qdrant_client.delete_collection(collection_name=collection_name)
        elif self.collection_name:
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
        else:
            raise CollectionNameNotProvidedException("Collection name not provided")
        if self.print_logs:
            print(f"Collection {collection_name} deleted successfully")
        self.collection_name = None


    def delete_data(
            self,
            filter: types.PointsSelector = None,
            collection_name: str = None
        ) -> None:
        '''
            Delete data from the collection

            Args:
                ids (List[int]): List of ids to delete
                collection_name (str, optional): Name of the collection. Defaults to None.
        '''
        if collection_name:
            self.qdrant_client.delete(collection_name=collection_name, points_selector=filter)
        elif self.collection_name:
            self.qdrant_client.delete(collection_name=self.collection_name, points_selector=filter)
        else:
            raise CollectionNameNotProvidedException("Collection name not provided")

        if self.print_logs:
            print(f"Data deleted successfully")

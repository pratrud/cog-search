"""Wrapper around Azure Cognitive Search."""
from __future__ import annotations

from azure.search.documents.indexes.models import (
    CorsOptions,
    ScoringProfile,
    SearchField,
    SemanticSearch,
    VectorSearch,
    VectorSearchProfile
)
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

import base64
import json
import logging
import uuid
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_env
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger()

# Allow overriding field names for Azure Search
FIELDS_ID = get_from_env(
    key="AZURESEARCH_FIELDS_ID", env_key="AZURESEARCH_FIELDS_ID", default="id"
)
FIELDS_CONTENT = get_from_env(
    key="AZURESEARCH_FIELDS_CONTENT",
    env_key="AZURESEARCH_FIELDS_CONTENT",
    default="content",
)
FIELDS_CONTENT_VECTOR = get_from_env(
    key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
    env_key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
    default="content_vector",
)
FIELDS_METADATA = get_from_env(
    key="AZURESEARCH_FIELDS_TAG", env_key="AZURESEARCH_FIELDS_TAG", default="metadata"
)

MAX_UPLOAD_BATCH_SIZE = 1000

def _get_search_client(
    endpoint: str,
    key: str,
    index_name: str,
    semantic_configuration_name: Optional[str] = None,
    fields: Optional[List[SearchField]] = None,
    vector_search: Optional[VectorSearch] = None,
    semantic_settings: Optional[SemanticSearch] = None,
    scoring_profiles: Optional[List[ScoringProfile]] = None,
    default_scoring_profile: Optional[str] = None,
    default_fields: Optional[List[SearchField]] = None,
    user_agent: Optional[str] = "langchain",
    cors_options: Optional[CorsOptions] = None,
) -> SearchClient:
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        VectorSearch,

        HnswAlgorithmConfiguration,
        SemanticPrioritizedFields,
        SemanticSearch
    )

    default_fields = default_fields or []
    if key is None:
        credential = DefaultAzureCredential()
    else:
        credential = AzureKeyCredential(key)
    index_client: SearchIndexClient = SearchIndexClient(
        endpoint=endpoint, credential=credential, user_agent=user_agent
    )

    try:
        index_client.get_index(name=index_name)
    except ResourceNotFoundError:
        # Fields configuration
        if fields is not None:
            # Check mandatory fields
            fields_types = {f.name: f.type for f in fields}
            mandatory_fields = {df.name: df.type for df in default_fields}
            # Check for missing keys
            missing_fields = {
                key: mandatory_fields[key]
                for key, value in set(mandatory_fields.items())
                - set(fields_types.items())
            }
            if len(missing_fields) > 0:
                # Helper for formatting field information for each missing field.
                def fmt_err(x: str) -> str:
                    return (
                        f"{x} current type: '{fields_types.get(x, 'MISSING')}'. "
                        f"It has to be '{mandatory_fields.get(x)}' or you can point "
                        f"to a different '{mandatory_fields.get(x)}' field name by "
                        f"using the env variable 'AZURESEARCH_FIELDS_{x.upper()}'"
                    )

                error = "\n".join([fmt_err(x) for x in missing_fields])
                raise ValueError(
                    f"You need to specify at least the following fields "
                    f"{missing_fields} or provide alternative field names in the env "
                    f"variables.\n\n{error}"
                )
        else:
            fields = default_fields
        # Vector search configuration
        if vector_search is None:
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default",
                        kind="hnsw",
                        parameters={  # type: ignore
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine",
                        },
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default",
                        kind="default"
                    )
                ]
            )

        # Create the semantic settings with the configuration
        if semantic_settings is None and semantic_configuration_name is not None:
            semantic_settings = SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name=semantic_configuration_name,
                        prioritized_fields=SemanticPrioritizedFields(
                            content_fields=[
                                SemanticField(field_name=FIELDS_CONTENT)
                            ],
                        ),
                    )
                ]
            )

        # Create the search index with the semantic settings and vector search
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_settings,
            scoring_profiles=scoring_profiles,
            default_scoring_profile=default_scoring_profile,
            cors_options=cors_options,
        )
        index_client.create_index(index)
    # Create the search client
    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=credential,
        user_agent=user_agent,
    )

class AzureCognitiveSearch(VectorStore):
    """Azure Cognitive Search vector store."""

    def __init__(
        self,
        azure_search_endpoint: str,
        azure_search_key: str,
        index_name: str,
        embedding_function: Embeddings,
        search_type: str = "hybrid",
        semantic_configuration_name: Optional[str] = None,
        semantic_query_language: str = "en-us",
        fields: Optional[List[SearchField]] = None,
        vector_search: Optional[VectorSearch] = None,
        semantic_settings: Optional[SemanticSearch] = None,
        scoring_profiles: Optional[List[ScoringProfile]] = None,
        default_scoring_profile: Optional[str] = None,
        **kwargs: Any,
    ):
        from azure.search.documents.indexes.models import (
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SimpleField,
        )

        """Initialize with necessary components."""
        # Initialize base class
        self.embedding_function = embedding_function
        default_fields = []
        if not fields:
            # only initializing default fields if fields not set
            # this avoids calling embed query an extra time
            default_fields = [
                SimpleField(
                    name=FIELDS_ID,
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                ),
                SearchableField(
                    name=FIELDS_CONTENT,
                    type=SearchFieldDataType.String,
                ),
                SearchField(
                    name=FIELDS_CONTENT_VECTOR,
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=len(embedding_function.embed_query("Text")),
                    vector_search_profile_name="default",
                ),
                SearchableField(
                    name=FIELDS_METADATA,
                    type=SearchFieldDataType.String,
                ),
            ]
        self.client = _get_search_client(
            azure_search_endpoint,
            azure_search_key,
            index_name,
            semantic_configuration_name=semantic_configuration_name,
            fields=fields,
            vector_search=vector_search,
            semantic_settings=semantic_settings,
            scoring_profiles=scoring_profiles,
            default_scoring_profile=default_scoring_profile,
            default_fields=default_fields,
        )
        self.search_type = search_type
        self.semantic_configuration_name = semantic_configuration_name
        self.semantic_query_language = semantic_query_language
        self.fields = fields if fields else default_fields

    @property
    def embeddings(self) -> Optional[Embeddings]:
        # TODO: Support embedding object directly
        return None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts data to an existing index."""
        keys = kwargs.get("keys")
        ids = []
        # Write data to index
        data = []

        texts = list(texts)

        try:
            embeddings = self.embedding_function.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self.embedding_function.embed_query(x) for x in texts]

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Use provided key otherwise use default key
            key = keys[i] if keys else str(uuid.uuid4())
            # Encoding key for Azure Search valid characters
            key = base64.urlsafe_b64encode(bytes(key, "utf-8")).decode("ascii")
            metadata = metadatas[i] if metadatas else {}
            # Add data to index
            # Additional metadata to fields mapping
            doc = {
                "@search.action": "upload",
                FIELDS_ID: key,
                FIELDS_CONTENT: text,
                FIELDS_CONTENT_VECTOR: np.array(
                    embedding, dtype=np.float32
                ).tolist(),
                FIELDS_METADATA: json.dumps(metadata),
            }
            if metadata:
                additional_fields = {
                    k: v
                    for k, v in metadata.items()
                    if k in [x.name for x in self.fields]
                }
                doc.update(additional_fields)
            data.append(doc)
            ids.append(key)
            # Upload data in batches
            if len(data) == MAX_UPLOAD_BATCH_SIZE:
                response = self.client.upload_documents(documents=data)
                # Check if all documents were successfully uploaded
                if not all([r.succeeded for r in response]):
                    raise Exception(response)
                # Reset data
                data = []

        # Considering case where data is an exact multiple of batch-size entries
        if len(data) == 0:
            return ids

        # Upload data to index
        response = self.client.upload_documents(documents=data)
        # Check if all documents were successfully uploaded
        if all([r.succeeded for r in response]):
            return ids
        else:
            raise Exception(response)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        search_type = kwargs.get("search_type", self.search_type)
        if search_type == "similarity":
            docs = self.vector_search(query, k=k, **kwargs)
        elif search_type == "hybrid":
            docs = self.hybrid_search(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            docs = self.semantic_hybrid_search(query, k=k, **kwargs)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")
        return docs

    def vector_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.vector_search_with_score(
            query, k=k, filters=kwargs.get("filters", None)
        )
        return [doc for doc, _ in docs_and_scores]

    def vector_search_with_score(
        self, query: str, k: int = 4, filters: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        results = self.client.search(
            search_text="",
            vector_queries=[
                VectorizedQuery(
                    vector=np.array(self.embedding_function.embed_query(query), dtype=np.float32).tolist(),
                    k_nearest_neighbors=k,
                    fields=FIELDS_CONTENT_VECTOR
                )
            ],
            select=[FIELDS_CONTENT, FIELDS_METADATA],
            filter=filters,
        )
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result[FIELDS_CONTENT],
                    metadata=json.loads(result[FIELDS_METADATA]),
                ),
                float(result["@search.score"]),
            )
            for result in results
        ]
        return docs

    def hybrid_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.hybrid_search_with_score(
            query, k=k, filters=kwargs.get("filters", None)
        )
        return [doc for doc, _ in docs_and_scores]

    def hybrid_search_with_score(
        self, query: str, k: int = 4, filters: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with an hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """

        results = self.client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(
                    vector=np.array(self.embedding_function.embed_query(query), dtype=np.float32).tolist(),
                    k_nearest_neighbors=k,
                    fields=FIELDS_CONTENT_VECTOR
                )
            ],
            select=[FIELDS_CONTENT, FIELDS_METADATA],
            filter=filters,
            top=k,
        )
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result[FIELDS_CONTENT],
                    metadata=json.loads(result[FIELDS_METADATA]),
                ),
                float(result["@search.score"]),
            )
            for result in results
        ]
        return docs

    def semantic_hybrid_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.semantic_hybrid_search_with_score(
            query, k=k, filters=kwargs.get("filters", None)
        )
        return [doc for doc, _, _ in docs_and_scores]

    def semantic_hybrid_search_with_score(
        self, query: str, k: int = 4, filters: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with an hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        results = self.client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(
                    vector=np.array(self.embedding_function.embed_query(query), dtype=np.float32).tolist(),
                    k_nearest_neighbors=50,
                    fields=FIELDS_CONTENT_VECTOR
                )
            ],
            select=[FIELDS_CONTENT, FIELDS_METADATA],
            filter=filters,
            query_type="semantic",
            semantic_configuration_name=self.semantic_configuration_name,
            query_caption="extractive",
            query_answer="extractive",
            top=k,
        )
        # Get Semantic Answers
        semantic_answers = results.get_answers() or []
        semantic_answers_dict: Dict = {}
        for semantic_answer in semantic_answers:
            semantic_answers_dict[semantic_answer.key] = {
                "text": semantic_answer.text,
                "highlights": semantic_answer.highlights,
            }
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result["content"],
                    metadata={
                        **json.loads(result["metadata"]),
                        **{
                            "captions": {
                                "text": result.get("@search.captions", [{}])[0].text,
                                "highlights": result.get("@search.captions", [{}])[
                                    0
                                ].highlights,
                            }
                            if result.get("@search.captions")
                            else {},
                            "answers": semantic_answers_dict.get(
                                json.loads(result["metadata"]).get("key"), ""
                            ),
                        },
                    },
                ),
                float(result["@search.score"]),
                float(result["@search.reranker_score"]),
            )
            for result in results
        ]

        return docs

    @classmethod
    def from_texts(
        cls: Type[AzureCognitiveSearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        index_name: str = "langchain-index",
        **kwargs: Any,
    ) -> AzureCognitiveSearch:
        # Creating a new Azure Search instance
        azure_search = cls(
            azure_search_endpoint,
            azure_search_key,
            index_name,
            embedding.embed_query,
        )
        azure_search.add_texts(texts, metadatas, **kwargs)
        return azure_search

if __name__ == "__main__":
    from config.azure_search_config import config
    from models.config.azure_cognitive_search_index_definition import AzureCognitiveSearchIndexDefinition
    from vectorstores.azure_cognitive_search_manager import AzureCognitiveSearchManager
    from langchain.embeddings import FakeEmbeddings

    domain_config = {
        "_id": "<auto-populated>",
        "domain": "<auto-populated>",
        "version": "<auto-populated>",
        "index_definition": {
        "fields": [
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "filterable": True
            },
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "content_vector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "vector_search_configuration": "test"
            },
            {
                "name": "metadata",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "source",
                "type": "Edm.String",
                "filterable": True
            },
            {
                "name": "ArticleSummary",
                "type": "Edm.String"
            },
            {
                "name": "Title",
                "type": "Edm.String"
            },
            {
                "name": "Category1",
                "type": "Edm.String"
            },
            {
                "name": "Category2",
                "type": "Edm.String"
            },
            {
                "name": "Category3",
                "type": "Edm.String"
            }
            ],
            "semantic_configuration_name": "semantic_configuration_with_categories_summary_and_title",
            "vector_search": {
                "algorithm_configurations": [
                    {
                        "name": "test",
                        "kind": "hnsw",
                        "hnsw_parameters": {
                            "metric": "cosine",
                            "m": 8,
                            "ef_construction": 600,
                            "ef_search": 700
                        }
                    }
                ]
            },
            "semantic_settings": {
            "configuration": [
                {
                "name": "default_semantic_configuration",
                "prioritized_fields": {
                    "prioritized_content_fields": [
                    {
                        "field_name": "content"
                    }
                    ]
                }
                },
                {
                "name": "semantic_configuration_with_categories_summary_and_title",
                "prioritized_fields": {
                    "title_field": {
                    "field_name": "Title"
                    },
                    "prioritized_content_fields": [
                    {
                        "field_name": "content"
                    },
                    {
                        "field_name": "ArticleSummary"
                    }
                    ],
                    "prioritized_keywords_fields": [
                    {
                        "field_name": "Category1"
                    },
                    {
                        "field_name": "Category2"
                    },
                    {
                        "field_name": "Category3"
                    }
                    ]
                }
                }
            ]
            }
        },
        "embedding_model_class": "TransformerEmbeddings",
        "embedding_class_args": {
        "tokenizer_path": "tokenizer-msmarco-bert-base-dpr-v5-updated",
        "document_embedding_path": "onnx-msmarco-bert-base-dot-v5",
        "query_embedding_path": "onnx-msmarco-bert-base-dpr-v5"
        },
        "file_loader_class_name": "JSONLoaderWithHtml",
        "splitter_class_name": "RecursiveCharacterTextSplitter",
        "splitter_kwargs": {
        "chunk_size": 1500,
        "chunk_overlap": 375
        },
        "llm_class_name": "AzureChatOpenAI",
        "ai_threshold": 0,
        "llm_kwargs": {
        "deployment_name": "gpt-35-turbo-16k",
        "temperature": 0,
        "max_tokens": 3000,
        "verbose": True,
        "model_version": "0613"
        },
        "chain_type": "stuff",
        "loader_kwargs": {
        "content_key": "BW_Article_Details__c",
        "jq_schema": ".",
        "metadata_func": "care_metadata",
        "text_content": False
        },
        "k_milvus": 20,
        "search_db_type": "azure_cognitive_search",
        "prompt": "You are a helpful AI assistant whose primary goal is to help AT&T call support agents who support calls from AT&T enterprise customers. According to, \n Context: {context}, \n what is the answer to the \n Question: {question}. \n Provide step by step instructions if available. Do not attempt to answer if the Context provided is empty. Ask them to elaborate the question instead."
    }

    collection_name = "c-user-pb5253-dev-2023-12-16"

    from azure.search.documents.indexes import SearchIndexClient
    from azure.core.credentials import AzureKeyCredential

    index_client = SearchIndexClient(endpoint=config.endpoint, credential=AzureKeyCredential(config.key))
    
    try:
        index_client.delete_index(collection_name)
    except:
        pass

    index_definition = AzureCognitiveSearchIndexDefinition(**domain_config["index_definition"])
    vector_search = index_definition.get_azure_vector_search()
    semantic_settings = index_definition.get_azure_semantic_settings()
    fields = index_definition.get_azure_fields()
    vector_store = AzureCognitiveSearchManager(
        azure_search_endpoint=config.endpoint,
        azure_search_key=config.key,
        embedding_function=FakeEmbeddings(size=768),
        fields=fields,
        index_name=collection_name,
        semantic_configuration_name=index_definition.semantic_configuration_name,
        vector_search=vector_search,
        search_type=index_definition.search_type,
        semantic_settings=semantic_settings
    )

    print(json.dumps(vector_store._index_client.get_index(collection_name).semantic_search.as_dict(), indent=2))
    print("\n\n")
    print(json.dumps([field.as_dict() for field in vector_store._index_client.get_index(collection_name).fields], indent=2))
    print("\n\n")
    print(json.dumps(vector_store._index_client.get_index(collection_name).vector_search.as_dict()))
    
    # try:
    #     collection = vector_store._index_client.get_index(collection_name)
    #     print(collection.semantic_search.as_dict())
    # finally:
    #     vector_store._index_client.delete_index(collection_name)

    try:
        vector_store.vector_store.add_texts([f"test_{i}" for i in range(100)])
        import time
        time.sleep(2)
        print("\n\n")
        print(len(vector_store.vector_store.hybrid_search('test', 10)))
        print(len(vector_store.vector_store.semantic_hybrid_search("test", 10)))
        print(len(vector_store.vector_store.vector_search("test", 10)))
    finally:
        # vector_store._index_client.delete_index(collection_name)
        print("done")
    # result = vector_store.search_with_score("test")
    # print(result)
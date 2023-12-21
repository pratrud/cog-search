from typing import (
    Any,
    List,
    Optional,
    Union
)

from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from loguru import logger as log
from .azure_cognitive_search import (
    AzureCognitiveSearch,
    FIELDS_CONTENT_VECTOR,
    FIELDS_ID,
)

from models.config.azure_cognitive_search_index_definition import (
    _SEMANTIC_HYBRID_SEARCH_TYPE
)

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from azure.search.documents.indexes.models import (
    ScoringProfile,
    SearchField,
    SemanticSearch,
    VectorSearch
)

_FIELDS_SOURCE = "source"
_FIELDS_USERID = "user_id"
_FIELDS_DOCUMENTID = "document_id"

def _get_search_index_client(endpoint: str, key: Union[str, None]):
    if key is None:
        credential = DefaultAzureCredential()
    else:
        credential = AzureKeyCredential(key)
    index_client: SearchIndexClient = SearchIndexClient(
        endpoint=endpoint, credential=credential, user_agent="langchain"
    )
    return index_client


class AzureCognitiveSearchManager:
    _vector_store: Union[AzureCognitiveSearch, None] = None
    _index_client: SearchIndexClient
    _index_name: str
    _search_type: str

    def __init__(
        self,
        azure_search_endpoint: str,
        azure_search_key: str,
        index_name: str,
        embedding_function: Embeddings,
        fields: List[SearchField],
        search_type: str = "hybrid",
        semantic_configuration_name: Optional[str] = None,
        semantic_query_language: str = "en-us",
        vector_search: Optional[VectorSearch] = None,
        semantic_settings: Optional[SemanticSearch] = None,
        scoring_profiles: Optional[List[ScoringProfile]] = None,
        default_scoring_profile: Optional[str] = None,
        **kwargs: Any,
    ):
        self._index_client = _get_search_index_client(
            azure_search_endpoint,
            azure_search_key
        )
        index_exists = self._does_index_exists(index_name)
        if index_exists:
            fields = self._index_client.get_index(index_name).fields
        else:
            content_vector_field = [field for field in fields if field.name == FIELDS_CONTENT_VECTOR][0]
            if not content_vector_field.vector_search_dimensions:
                content_vector_field.vector_search_dimensions = len(embedding_function.embed_query("Text"))
            if not content_vector_field.vector_search_profile_name:
                content_vector_field.vector_search_profile_name = "default"

        self._vector_store_args = {
            "azure_search_endpoint": azure_search_endpoint,
            "azure_search_key": azure_search_key,
            "index_name": index_name,
            "embedding_function": embedding_function,
            "search_type": search_type,
            "semantic_configuration_name": semantic_configuration_name,
            "semantic_query_language": semantic_query_language,
            "fields": fields,
            "vector_search": vector_search,
            "semantic_settings": semantic_settings,
            "scoring_profiles": scoring_profiles,
            "default_scoring_profile": default_scoring_profile,
            **kwargs
        }
        self._index_name = index_name
        self._vector_store = self._get_vector_store()

        self._search_type = search_type

    def _does_index_exists(self, index_name: str):
        try:
            self._index_client.get_index(index_name)
            return True
        except ResourceNotFoundError:
            return False


    def _get_vector_store(self):
        if self._vector_store:
            return self._vector_store
        return AzureCognitiveSearch(
            **self._vector_store_args
        )


    def delete_index(self):
        # only delete the index if the vector store exists (as the index exists)
        if not self._vector_store:
            return

        # NOTE: No exception is thrown if the index does not exists
        self._index_client.delete_index(self._index_name)
        self._vector_store = None


    def upload_documents(self, 
                         documents: List[Document], 
                         filter_attributes: dict = {},
                         drop_old: bool = False):
        # if drop old, delete the index and recreate the index
        if drop_old:
            self.delete_index()

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata|filter_attributes for d in documents]
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas
        )


    def search_with_score(self, query: str, k: int = 4, filters: Optional[str] = None, ai_threshold: float = 0):
        if self._search_type == _SEMANTIC_HYBRID_SEARCH_TYPE:
            log.debug("searching with the semantic hybrid configuration.")
            docs = self.vector_store.semantic_hybrid_search_with_score(query, k, filters)
            docs = [doc for doc in filter(lambda d: (d[2]>=ai_threshold), docs)]
            return docs

        log.debug("searching with the hybrid configuration.")
        return self.vector_store.hybrid_search_with_score(query, k, filters)

    def delete_by_source(self, source: str):
        search_client = self._index_client.get_search_client(self._index_name)
        documents = search_client.search(
            search_text="",
            filter=f"source eq '{source}'",
            select=[FIELDS_ID]
        )

        documents = list(documents)
        if len(documents) < 1:
            log.warning(f"No elements to delete from index {self._index_name} given the source {source}.")
            return

        search_client.delete_documents(documents)
        log.info(f"Deleted {len(documents)} from index {self._index_name} given the source  {source}.")

    def delete_by_id(self, document_id: str):
        search_client = self._index_client.get_search_client(self._index_name)
        documents = search_client.search(
            search_text="",
            filter=f"document_id eq '{document_id}'",
            select=[FIELDS_ID]
        )

        documents = list(documents)
        if len(documents) < 1:
            log.warning(f"No elements to delete from index {self._index_name} given the document_id {document_id}.")
            return

        search_client.delete_documents(documents)
        log.info(f"Deleted {len(documents)} from index {self._index_name} given the document_id  {document_id}.")

    def delete_by_source_filename(self, suffix: str):
        search_client = self._index_client.get_search_client(self._index_name)
        documents = search_client.search(
            search_text=""
        )
        documents = list(documents)
        if len(documents) < 1:
            log.warning(f"No elements to delete from index {self._index_name} given the source {suffix}.")
            return
        documents_to_delete = [doc for doc in documents if doc["source"].endswith(suffix)]
        search_client.delete_documents(documents_to_delete)
        log.info(f"Deleted {len(documents_to_delete)} from index {self._index_name} given the source  {suffix}.")

    def get_index_statistics(self):
        stats = self._index_client.get_index_statistics(self._index_name)
        return stats

    @property
    def vector_store(self) -> AzureCognitiveSearch:
        self._vector_store = self._get_vector_store()
        return self._vector_store

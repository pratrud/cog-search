import os
from typing import Any, Tuple, List, Dict
from time import time
from langchain.callbacks import get_openai_callback
from azure.search.documents.indexes.models import (
    SearchField,
    SemanticSearch,
    VectorSearch
)
from vectorstores.azure_cognitive_search_manager import AzureCognitiveSearchManager
from loguru import logger as log
from context import VectorDBUtil
from models.config.azure_cognitive_search_index_definition import AzureCognitiveSearchIndexDefinition
from config.azure_search_config import config as azure_search_config
from tenacity import retry, stop_after_attempt, wait_fixed

_DEFAULT_DOMAIN = "default"
_DEFAULT_VERSION = "default"

_RETRY_THRESHOLD = 3
_RETRY_WAIT_IN_SECONDS = 2


# we should place this in a util
def _create_pending_file_path(domain: str, version: str, file_name: str):
    return rf"./docstore/{domain}/{version}/pending/{file_name}"

def _create_pending_file_path_byod(domain: str, user_id: str, file_name: str):
    return rf"./docstore/{domain}/{user_id}/{file_name}"

def _convert_dict_to_azure_cognitive_search_index_definition(obj: dict):
    return AzureCognitiveSearchIndexDefinition(**obj)


class AzureCognitiveSearchDb():
    vector_store_manager: AzureCognitiveSearchManager
    domain: str
    domain_params: dict
    current_embedding: Any

    def __init__(self, domain: str, domain_params: dict, vdb_util: VectorDBUtil,is_byod:bool=False):
        super().__init__()
        self.domain = domain
        self.domain_params = domain_params
        self.version = domain_params.get("version", None)
        
        self.current_embedding = vdb_util.return_embedding_model()
        self._connection_args = {
            "endpoint": azure_search_config.endpoint,
            "key": azure_search_config.key
        }
        
        if is_byod:
            self.user_id = domain_params.get("user_id", None)
            self.collection_name = self.create_collection_name(self.domain or _DEFAULT_DOMAIN, os.environ.get("ENV"),
                                                                  self.user_id[:2])
        
        else:
            self.collection_name = self.create_collection_name(self.domain or _DEFAULT_DOMAIN, os.environ.get("ENV"),
                                                                               self.version or _DEFAULT_VERSION)

        log.info(
            f"Collection name to be used by domain [{domain}] is [{self.collection_name}]")
        log.info(f"Domain [{domain}] started on version [{self.version}]")
        self.vdb_util = vdb_util
        self.load_db()

    def load_db(self):
        try:
            index_definition = self.domain_params.get("index_definition", {})
            index_definition = _convert_dict_to_azure_cognitive_search_index_definition(
                index_definition)
            vector_search = index_definition.get_azure_vector_search()
            semantic_settings = index_definition.get_azure_semantic_settings()
            fields = index_definition.get_azure_fields()
            log.debug(f"index definition: {index_definition.json()}")
        except Exception as e:
            log.error(
                f"For client: [{self.domain}] - {e}"
            )
            raise Exception(
                f"There were errors parsing or transforming the index definition. {e}")

        try:
            self._load_db(
                fields,
                index_definition.semantic_configuration_name,
                vector_search,
                index_definition.search_type,
                semantic_settings
            )
        except Exception as e:
            log.error(
                f"For client: [{self.domain}] - {e}"
            )
           
        finally:
            # recycle the connection args once they are used
            self._connection_args = {}

    @retry(
        reraise=True,
        stop=stop_after_attempt(_RETRY_THRESHOLD),
        wait=wait_fixed(_RETRY_WAIT_IN_SECONDS)
    )
    def _load_db(
            self,
            fields: List[SearchField],
            semantic_configuration_name: str | None,
            vector_search: VectorSearch,
            search_type: str,
            semantic_settings: SemanticSearch | None):
        self.vector_store_manager = AzureCognitiveSearchManager(
            azure_search_endpoint=self._connection_args.get("endpoint"),
            azure_search_key=self._connection_args.get("key"),
            embedding_function=self.current_embedding,
            fields=fields,
            index_name=self.collection_name,
            semantic_configuration_name=semantic_configuration_name,
            vector_search=vector_search,
            search_type=search_type,
            semantic_settings=semantic_settings
        )

    def create_collection_name(self, domain: str, environment: str, version: str):
        return f'c-{domain}-{environment}-{version}'.replace(
            "_", "-").lower()


    def upload_single_document_byod(self, file_name:str, filter_params:dict):
        user_id = filter_params["user_id"]
        base_location = fr"./docstore/byod/{user_id}"
        log.info("Starting loading single document...")
        chunked_documents, self.current_embedding = self.vdb_util.ingest_single_byod(f"{base_location}/{file_name}", filter_params)
        log.info("Single document loaded. Starting embeddings and load to azure cognitive search")
        self.vector_store_manager.upload_documents(chunked_documents, filter_params, False)
        log.info("Single document embedded and loaded to azure cognitive search.")

    def upload_single_document(self, file_name: str = None, filter_attributes: dict = {}):
        log.info("Starting loading single document...")
        pending_file_path = _create_pending_file_path(self.domain, self.version, file_name)
        chunked_documents, self.current_embedding = self.vdb_util.ingest_single(pending_file_path)
        log.info("Single document loaded. Starting embeddings and load to azure cognitive search")
        self.vector_store_manager.upload_documents(chunked_documents, filter_attributes, False)
        log.info("Single document embedded and loaded to azure cognitive search.")

    def chat_query(self, query: str, 
                   chat_history: List[Tuple[str, Any]] = None, 
                   override_prompt: str = None, 
                   context: str = None, 
                   useHistory: str = None, 
                   query_filter: Dict[str, List[str]] = None
                   ):
        result = self.chat_query_details(query, chat_history, override_prompt, context, useHistory, query_filter)
        return result
    
    def chat_query_byod(self, user_id: str, document_id: str, query: str, chat_history: List[Tuple[str, Any]] = None, override_prompt: str = None):

        llm = self.vdb_util.language_model(**self.domain_params.get("llm_kwargs", {}))
        if len(document_id):
            document_ids_expr = " or ".join([f"document_id eq '{i}'" for i in document_id]) + f" and user_id eq '{user_id}'"
            
        if chat_history is None:
            chat_history = []

        if self.domain_params.get("reranker_class_name", None) is None:

            start_similarity_search = time()
            similarity_res = self.similarity_search_with_score(query=query,
                                                               k=self.domain_params.get("k_milvus", 10), filters=document_ids_expr,
                                                               ai_threshold=self.domain_params.get("ai_threshold", 0))
            similarity_res.sort(key=lambda tup: tup[1], reverse=True)
            elapsed_similarity_search = (time() - start_similarity_search) 

            start_build_context = time()
            context_list = [entry[0].page_content for entry in similarity_res]

            context = ""
            for context_item in context_list:
                context += context_item + "\n"

            if override_prompt is not None:
                prompt = override_prompt
            else:
                prompt = self.domain_params.get("prompt", 'DEFAULT_PROMPT')

            prompt_final = prompt.format(question=query, context=context)
            if len(chat_history) > 0:
                prompt_final = prompt.format(question=query, context=context, chat_history=chat_history)
            elapsed_build_context = (time() - start_build_context)
            with get_openai_callback() as cb:
                start_llm_call = time()
                result = llm.predict(prompt_final)
                usage = cb.__dict__
                elapsed_llm_call = (time() - start_llm_call)
            usage["similarity_rearch_seconds"] = elapsed_similarity_search
            usage["build_context_seconds"] = elapsed_build_context
            usage["llm_call_seconds"] = elapsed_llm_call
            
            # TODO: return LLM response from Langchain


        else:
            start_similarity_search = time()
            similarity_res = self.similarity_search_with_score(query=query,
                                                               k=self.domain_params.get("k_milvus", 10), filters=document_ids_expr,
                                                               ai_threshold=self.domain_params.get("ai_threshold", 0))
            similarity_res.sort(key=lambda tup: tup[1], reverse=True)
            elapsed_similarity_search = (time() - start_similarity_search)
            ranked = []
            start_build_context = time()
            for kv in similarity_res:
                ranked.append([kv[0], self.vdb_util.reranker.predict([query, kv[0].page_content])])

            ranked.sort(key=lambda tup: tup[1], reverse=True)
            top_ranked = ranked[:self.domain_params.get("k_reranker", 3)]
            context_list = [entry[0].page_content for entry in top_ranked]

            context = ""
            for idx, context_item in enumerate(context_list):
                context += context_item + "\n"

            if override_prompt is not None:
                prompt = override_prompt
            else:
                prompt = self.domain_params.get("prompt", 'DEFAULT_PROMPT')
            final_prompt = prompt.format(question=query, context=context)
            if len(chat_history) > 0:
                final_prompt = prompt.format(question=query, context=context, chat_history=chat_history)

            elapsed_build_context = (time() - start_build_context)
            with get_openai_callback() as cb:
                start_llm_call = time()
                result = llm.predict(final_prompt)
                usage = cb.__dict__
                elapsed_llm_call = (time() - start_llm_call) 
            usage["similarity_rearch_seconds"] = elapsed_similarity_search
            usage["build_context_seconds"] = elapsed_build_context
            usage["llm_call_seconds"] = elapsed_llm_call
            # TODO: return LLM response from Langchain

    def upload_documents(self, document_location: str = None, drop_old: bool = False):
        # if uploaded in the document location, we need to drop old index
        drop_old = True if "uploaded" in str(document_location) else False

        # add documents to the index
        chunked_documents, _ = self.vdb_util.ingest_multi(document_location)
        self.vector_store_manager.upload_documents(chunked_documents, drop_old)
        log.info("Documents embedded and loaded to Azure Cognitive Search.")

    def similarity_search_with_score(self, query: str, k: int = 4, filters: str = None, ai_threshold: float = 0):
        return self.vector_store_manager.search_with_score(query, k, filters, ai_threshold=ai_threshold)

    def delete_document_by_field(self, domain: str, config_version: str, file_name: str):
        source = _create_pending_file_path(domain, config_version, file_name)
        self.vector_store_manager.delete_by_source(source)
        log.info(
            f"{source} deleted matching with the collection name [{self.collection_name}]")

    def delete_document_by_field_byod(self, domain:str, user_id:str, file_name:str):
        source = _create_pending_file_path_byod(domain, user_id, file_name)
        self.vector_store_manager.delete_by_source(source)
        log.info(f"{source} deleted matching with the collection name [{self.collection_name}]")
    
    def delete_document_by_field_byod_id(self, document_id:str):
        self.vector_store_manager.delete_by_id(document_id=document_id)
        log.info(f"{document_id} deleted matching with the collection name [{self.collection_name}]")

    def delete_data_by_fieldname(self, field_name:str, file_name:str):
        self.vector_store_manager.delete_by_source_filename(file_name)
        log.info(
            f"{file_name} deleted matching with the collection name [{self.collection_name}]")

    def drop_collection(self):
        self.vector_store_manager.delete_index()
        log.info(
            rf"Dropped collection {self.collection_name} from Azure Cognitive Search")

    def stats_uploaded_vectorstore(self):
        try:
            stats = self.vector_store_manager.get_index_statistics()
            stats["Collection name"] = self.collection_name
            stats['storage_size_bytes'] = stats.pop('storage_size')
            return stats

        except Exception as e:
            log.error(e)
            log.error(
                rf"Unable to fetch stats from vectorstore for domain {self.domain}.")

    def create_collection(self, schema_body):
        return {"message": "API is not yet implemented to support cognitive search"}

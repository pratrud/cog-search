import abc
from pydantic import BaseModel, validator
from typing import Optional, List, Generic, TypeVar
from azure.search.documents.indexes.models import (
    HnswParameters as AzureHnswParameters,
    SemanticConfiguration,
    VectorSearch as AzureVectorSearch,
    HnswAlgorithmConfiguration as AzureVectorSearchAlgorithmConfiguration,
    SemanticPrioritizedFields as AzurePrioritizedFields,
    SemanticConfiguration as AzureSemanticConfiguration,
    SemanticSearch as AzureSemanticSettings,
    SemanticField as AzureSemanticField,
    SearchField,
    SearchFieldDataType,
    VectorSearchProfile as AzureVectorSearchProfile
)
from loguru import logger as log

from vectorstores.azure_cognitive_search import (
    FIELDS_CONTENT,
    FIELDS_CONTENT_VECTOR,
    FIELDS_ID,
    FIELDS_METADATA
)

# index types
_HNSW_INDEX_TYPE = "hnsw"

# metric types
_COSINE_METRIC_TYPE = "cosine"
_EUCLIDEAN_METRIC_TYPE = "euclidean"
_DOT_PRODUCT_METRIC_TYPE = "dotProduct"

_METRIC_TYPES = [
    _COSINE_METRIC_TYPE,
    _DOT_PRODUCT_METRIC_TYPE,
    _EUCLIDEAN_METRIC_TYPE,
]

# search types
_HYBRID_SEARCH_TYPE = "hybrid"
_SEMANTIC_HYBRID_SEARCH_TYPE = "semantic_hybrid"

# params
_DEFAULT_M = 4
_MIN_M = 4
_MAX_M = 10

_DEFAULT_EF_SEARCH = 500
_MIN_EF_SEARCH = 100
_MAX_EF_SEARCH = 1000

_DEFAULT_EF_CONSTRUCTION = 400
_MIN_EF_CONSTRUCTION = 100
_MAX_EF_CONSTRUCTION = 1000

# other defaults
_DEFAULT_ALGORITHM_CONFIGURATION_NAME = "default"

# required field names
FIELDS_SOURCE = "source"
_REQUIRED_FIELDS = [
    FIELDS_ID,
    FIELDS_CONTENT,
    FIELDS_CONTENT_VECTOR,
    FIELDS_METADATA,
    FIELDS_SOURCE,
]


T = TypeVar('T')
class _BaseAzure(Generic[T], BaseModel, abc.ABC):
    @abc.abstractmethod
    def to_azure(self) -> T:
        return


class HnswParameters(BaseModel):
    metric: str = _COSINE_METRIC_TYPE
    m: int = _DEFAULT_M
    ef_construction: int = _DEFAULT_EF_CONSTRUCTION
    ef_search: int = _DEFAULT_EF_SEARCH

    @validator('m')
    @classmethod
    def validate_m(cls, m: int) -> int:
        if m > _MAX_M or m < _MIN_M:
            log.warning(f"m ({m}) is out of range [{_MIN_M}, {_MAX_M}]. Using the default m of {_DEFAULT_M}.")
            return _DEFAULT_M
        return m
    
    @validator('ef_construction')
    @classmethod
    def validate_ef_construction(cls, ef_construction: int) -> int:
        if ef_construction > _MAX_EF_CONSTRUCTION or ef_construction < _MIN_EF_CONSTRUCTION:
            log.warning(f"ef construction ({ef_construction}) is out of range [{_MIN_EF_CONSTRUCTION}, {_MAX_EF_CONSTRUCTION}]. Using the default ef construction of {_DEFAULT_EF_CONSTRUCTION}.")
            return _DEFAULT_EF_CONSTRUCTION
        return ef_construction
    
    @validator('ef_search')
    @classmethod
    def validate_ef_search(cls, ef_search: int) -> int:
        if ef_search > _MAX_EF_SEARCH or ef_search < _MIN_EF_SEARCH:
            log.warning(f"ef search ({ef_search}) is out of range [{_MIN_EF_SEARCH}, {_MAX_EF_SEARCH}]. Using the default ef search of {_DEFAULT_EF_SEARCH}.")
            return _DEFAULT_EF_SEARCH
        return ef_search
    
    @validator('metric')
    @classmethod
    def validate_metric_type(cls, metric_type: str) -> str:
        lowered_metric_types = [mt.lower() for mt in _METRIC_TYPES]
        lowered_metric_type = metric_type.lower()
        if lowered_metric_type not in lowered_metric_types:
            log.warning(f"metric ({metric_type}) is not in the supported metric list {_METRIC_TYPES}. Using the default metric of {_COSINE_METRIC_TYPE}.")
            return _COSINE_METRIC_TYPE
        
        # get the index where the values are the same
        i = lowered_metric_types.index(lowered_metric_type)
        return _METRIC_TYPES[i]


class AlgorithmConfiguration(BaseModel):
    name: str = _DEFAULT_ALGORITHM_CONFIGURATION_NAME
    hnsw_parameters: HnswParameters = HnswParameters()

    @property
    def kind(self):
        return _HNSW_INDEX_TYPE


class VectorSearch(BaseModel):
    algorithm_configurations: Optional[List[AlgorithmConfiguration]] = [AlgorithmConfiguration()]

    @validator('algorithm_configurations')
    @classmethod
    def validate_algorithm_configurations(cls, algorithm_configurations):
        if not algorithm_configurations:
            return [AlgorithmConfiguration()]
        return algorithm_configurations
    
class Field(_BaseAzure[SearchField]):
    name: str
    type: str
    searchable: bool = False
    filterable: bool = False
    sortable: bool = False
    facetable: bool = False
    key: bool = False
    retrievable: bool = True
    vector_search_dimensions: Optional[int] = None
    vector_search_configuration: Optional[str] = None

    def to_azure(self) -> SearchField:
        return SearchField(
            name=self.name,
            type=self.type,
            key=self.key,
            searchable=self.searchable,
            filterable=self.filterable,
            sortable=self.sortable,
            facetable=self.facetable,
            vector_search_dimensions=self.vector_search_dimensions,
            vector_search_profile_name=self.vector_search_configuration,
        )

class SemanticField(_BaseAzure[AzureSemanticField]):
    field_name: str

    def to_azure(self) -> AzureSemanticField:
        return AzureSemanticField(
            field_name=self.field_name
        )

class PrioritizedFields(_BaseAzure[AzurePrioritizedFields]):
    title_field: Optional[SemanticField]
    prioritized_content_fields: Optional[List[SemanticField]]
    prioritized_keywords_fields: Optional[List[SemanticField]]

    def to_azure(self) -> AzurePrioritizedFields:
        title_field = None if not self.title_field else self.title_field.to_azure()
        prioritized_content_fields = (
            None
            if not self.prioritized_content_fields
            else [field.to_azure() for field in self.prioritized_content_fields]
        )
        prioritized_keywords_fields = (
            None
            if not self.prioritized_keywords_fields
            else [field.to_azure() for field in self.prioritized_keywords_fields]
        )
        return AzurePrioritizedFields(
            title_field=title_field,
            content_fields=prioritized_content_fields,
            keywords_fields=prioritized_keywords_fields
        )

class SemanticConfiguration(_BaseAzure[AzureSemanticConfiguration]):
    name: str
    prioritized_fields: PrioritizedFields

    def to_azure(self) -> AzureSemanticConfiguration:
        return AzureSemanticConfiguration(
            name=self.name,
            prioritized_fields=self.prioritized_fields.to_azure()
        )

class SemanticSettings(_BaseAzure[AzureSemanticSettings]):
    default_configuration: Optional[str]
    configuration: Optional[List[SemanticConfiguration]]

    def to_azure(self) -> AzureSemanticSettings:
        configuration = (
            None
            if not self.configuration
            else [config.to_azure() for config in self.configuration]
        )

        return AzureSemanticSettings(
           default_configuration_name=self.default_configuration,
           configurations=configuration
        )

class AzureCognitiveSearchIndexDefinition(BaseModel):
    semantic_configuration_name: Optional[str]
    semantic_settings: Optional[SemanticSettings]
    vector_search: VectorSearch = VectorSearch()
    fields: List[Field] = [
        Field(
            name=FIELDS_ID,
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        Field(
            name=FIELDS_CONTENT,
            type=SearchFieldDataType.String,
            searchable=True
        ),
        Field(
            name=FIELDS_CONTENT_VECTOR,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_configuration="default",
        ),
        Field(
            name=FIELDS_METADATA,
            type=SearchFieldDataType.String,
            searchable=True
        ),
        Field(
            name=FIELDS_SOURCE,
            type=SearchFieldDataType.String,
            filterable=True
        )
    ]

    @property
    def search_type(self):
        if self.semantic_configuration_name:
            return _SEMANTIC_HYBRID_SEARCH_TYPE
        return _HYBRID_SEARCH_TYPE
    
    @validator('fields')
    @classmethod
    def validate_fields(cls, fields: List[Field]):
        if not fields:
            return fields

        field_names = set([field.name for field in fields])
        field_name_intersection = field_names.intersection(_REQUIRED_FIELDS)
        if field_name_intersection != set(_REQUIRED_FIELDS):
            # using Exception over ValueError to bypass the pydantic error message formatter
            raise Exception(f"Missing the following required fields: {list(set(_REQUIRED_FIELDS).difference(field_name_intersection))}")
        return fields

    def get_azure_fields(self) -> List[SearchField]:
        return [field.to_azure() for field in self.fields]

    def get_azure_semantic_settings(self) -> Optional[AzureSemanticSettings]:
        if not self.semantic_settings:
            return None
        return self.semantic_settings.to_azure()

    def get_azure_vector_search(self):
        return AzureVectorSearch(
            algorithms=[
                AzureVectorSearchAlgorithmConfiguration(
                    name=self.vector_search.algorithm_configurations[0].name,
                    kind=self.vector_search.algorithm_configurations[0].kind,
                    parameters=AzureHnswParameters(
                        metric=self.vector_search.algorithm_configurations[0].hnsw_parameters.metric,
                        m=self.vector_search.algorithm_configurations[0].hnsw_parameters.m,
                        ef_construction=self.vector_search.algorithm_configurations[0].hnsw_parameters.ef_construction,
                        ef_search=self.vector_search.algorithm_configurations[0].hnsw_parameters.ef_search,
                    )
                ),
            ],
            profiles=[
                AzureVectorSearchProfile(
                    name=self.vector_search.algorithm_configurations[0].name,
                    algorithm_configuration_name=self.vector_search.algorithm_configurations[0].name
                )
            ]
        )

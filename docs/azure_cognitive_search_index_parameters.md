# Azure Cognitive Search Index Parameters <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [Overview](#overview)
- [Config Update](#config-update)
- [Appendix](#appendix)
  - [Required Fields](#required-fields)

## Overview

This document plans to outline the index definition parameters for Azure Cognitive Search.
There are a few important parameters that an admin can update to adjust the performance of Azure Cognitive Search.
Some of the important parameters that the admin can change include `m`, `ef_construction`, and `ef_search`.
The utilization of semantic hybrid search can also be enabled or disabled.

## Config Update

The Domain Services repository is driven by configuration defined by business units/domains.
One important part that the domain configuration can impact is the configuration of the search index (in this case, Azure Cognitive Search).
Azure Cognitive Search relies on the HNSW (Hierarchical Navigable Small World) approximate search configuration which has the following parameter:

- `ef_construction`  [DEFAULT=400;MIN=100;MAX=1000]: The size of the dynamic list containing the nearest neighbors, which is used during index time. Increasing this parameter may improve index quality, at the expense of increased indexing time. At a certain point, increasing this parameter leads to diminishing returns.
- `ef_search`  [DEFAULT=500;MIN=100;MAX=1000]: The size of the dynamic list containing the nearest neighbors, which is used during search time. Increasing this parameter may improve search results, at the expense of slower search. Increasing this parameter leads to diminishing returns.
- `m` [DEFAULT=4;MIN=4;MAX=10;]: The number of bi-directional links created for every new element during construction. Increasing this parameter value may improve recall and reduce retrieval times for datasets with high intrinsic dimensionality at the expense of increased memory consumption and longer indexing time.
- `metric` [DEFAULT=cosine;OPTIONS=(cosine,euclidean,dotProduct)]: The similarity metric to use for vector comparisons.

Azure Cognitive Search also allows for semantic hybrid, hybrid (vector search and simple text search), and vector search.
Semantic hybrid search leverages both vector search and Azure Cognitive Search's internal semantic search.
The product of the vector search and internal search go through a re-ranking process, and the final results are returned post re-ranking.
For more details on how semantic search works in Azure Cognitive Search, please review [this document](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview).

When leveraging Azure Cognitive Search, the configuration, in the configuration database, should be updated to reflect the following:

```json
{
  "_id": "<document-id>",
  ...
  "index_definition": {
    "semantic_configuration_name": "semantic_configuration",
    "semantic_settings": {
      "configuration": [
        {
          "name": "semantic_configuration",
          "prioritized_fields": {
            "prioritized_content_fields": [
                {
                    "field_name": "content"
                }
            ],
            "prioritized_keywords_fields": [
                {
                    "field_name": "<keyword-field>"
                }
            ],
            "title_field": { "field_name": "<title-field>" }
          }
        }
      ]
    },
    "vector_search": {
        "algorithm_configurations": [
            {
                "name": "default",
                "kind": "hnsw",
                "hnsw_parameters": {
                    "metric": "cosine",
                    "m": 4,
                    "ef_construction": 400,
                    "ef_search": 500
                }
            }
        ]
    }
  },
  "fields": [
    {
      "name": "id",
      "type": "Edm.String",
      "key": true,
      "filterable": true
    },
    {
      "name": "content",
      "type": "Edm.String",
      "searchable": true
    },
    {
      "name": "content_vector",
      "type": "Collection(Edm.Single)",
      "searchable": true,
      "vector_search_configuration": "default"
    },
    {
      "name": "metadata",
      "type": "Edm.String",
      "searchable": true
    },
    {
      "name": "source",
      "type": "Edm.String",
      "filterable": true
    }
  ],
  "k_milvus": 10,
  "k_reranker": 5,
  "search_db_type": "azure_cognitive_search",
}
```

Where:

- `index_definition.semantic_configuration_name` [DEFAULT=None]: The name of the semantic configuration. If specified, Azure Cognitive Search semantic hybrid search will be available when querying the index. If not specified, hybrid search is used.
- `index_definition.vector_search.algorithm_configurations[0].name` [DEFAULT=default]: Then name of the algorithm config.
- `index_definition.vector_search.algorithm_configurations[0].kind` [IMMUTABLE=hnsw]: The kind of ANN algorithm.
- `index_definition.vector_search.algorithm_configurations[0].hnsw_parameters.metric` [DEFAULT=cosine]: The metric used.
- `index_definition.vector_search.algorithm_configurations[0].hnsw_parameters.m` [DEFAULT=4]: The value for `m`.
- `index_definition.vector_search.algorithm_configurations[0].hnsw_parameters.ef_construction` [DEFAULT=400]: The value for `ef_construction`.
- `index_definition.vector_search.algorithm_configurations[0].hnsw_parameters.ef_search` [DEFAULT=500]: The value for `ef_search`.
- `k_milvus`: The number of results to return after searching the index. This field should be renamed in the future as it should not be dependent on the vector database.
- `k_reranker`: The number of results to return after re-ranking.
- `search_db_type` [DEFAULT=milvus]: The search database type.
- `index_definition.semantic_settings.configurations[*].name`: The name of the semantic configuration when leveraging the **semantic_settings**.
- `index_definition.semantic_settings.configurations[*].prioritized_fields.prioritized_content_fields[*].field_name`: Longer chunks of text in natural language form, subject to maximum token input limits (2000 tokens) on the machine learning models. Common examples include the body of a document, description of a product, or other free-form text.
- `index_definition.semantic_settings.configurations[*].prioritized_fields.prioritized_keywords_fields[*].field_name`: A list of keywords, such as the tags on a document, or a descriptive term, such as the category of an item.
- `index_definition.semantic_settings.configurations[*].prioritized_fields.title_field.field_name`: A short string, ideally under 25 words. This field could be the title of a document, name of a product, or a unique identifier. If you don't have suitable field, leave it blank.
- `index_definition.fields[*].name`: The name of the field.
- `index_definition.fields[*].type`: the type of the field.
- `index_definition.fields[*].searchable` [DEFAULT=False]: Full-text searchable, subject to lexical analysis such as word-breaking during indexing. If you set a searchable field to a value like "sunny day", internally it's split into the individual tokens "sunny" and "day".
- `index_definition.fields[*].filterable` [DEFAULT=False]: Referenced in $filter queries. Filterable fields of type Edm.String or Collection(Edm.String) don't undergo word-breaking, so comparisons are for exact matches only. For example, if you set such a field f to "sunny day", $filter=f eq 'sunny' finds no matches, but $filter=f eq 'sunny day' will.
- `index_definition.fields[*].sortable` [DEFAULT=False]: 	By default the system sorts results by score, but you can configure sort based on fields in the documents. Fields of type Collection(Edm.String) can't be "sortable".
- `index_definition.fields[*].facetable` [DEFAULT=False]: Typically used in a presentation of search results that includes a hit count by category (for example, hotels in a specific city). This option can't be used with fields of type Edm.GeographyPoint. Fields of type Edm.String that are filterable, "sortable", or "facetable" can be at most 32 kilobytes in length.
- `index_definition.fields[*].key` [DEFAULT=False]: Unique identifier for documents within the index. Exactly one field must be chosen as the key field and it must be of type `Edm.String`.
- `index_definition.fields[*].retrievable` [DEFAULT=True]: Determines whether the field can be returned in a search result. This is useful when you want to use a field (such as profit margin) as a filter, sorting, or scoring mechanism, but don't want the field to be visible to the end user. This attribute must be `true` for `key` fields.
- `index_definition.fields[*].vector_search_configuration` [DEFAULT=None]: The vector search configuration to use. The `vector_search_configuration` value should match the value of `index_definition.vector_search.algorithm_configurations[0].name` (which defaults to **default**).
- `index_definition.fields[*].vector_search_dimensions` [DEFAULT=None]: The dimensions of the vector. The dimensions attribute has a minimum of 2 and a maximum of 2048 floating point values each.

**NOTE(\*):** If the index definition contains `semantic_configuration_name` but no `semantic_settings`, `semantic_settings` will be created for you with the content field as the only field in `prioritized_content_fields`. More information on the `prioritized_content_fields` and Azure Cognitive Search semantic search can be found [here](https://learn.microsoft.com/en-us/azure/search/semantic-how-to-query-request?tabs=portal%2Cportal-query).

**NOTE(\*\*):** If an index is initialized without `semantic_configuration_name` and `semantic_settings`, the index cannot be updated to allow for semantic search (at this time).
In index can only toggle between semantic search and hybrid search if the index is created with `semantic_configuration_name` or `semantic_settings`.

**NOTE(\*\*\*):** `index_definition.fields` is an optional property.
If not set, the default fields are used. 
The default fields can be found in [the appendix](#required-fields).
When defining fields to use, there are several fields that are required.
The fields that are required are the following:

- id
- content
- content_vector
- metadata
- source

When defining a custom fields schema, it is recommended to start with the default fields and add the additional fields.
This ensures that all the required fields exist and are defined accordingly.

## Appendix

### Required Fields

```json
[
  {
    "name": "id",
    "type": "Edm.String",
    "key": true,
    "filterable": true
  },
  {
    "name": "content",
    "type": "Edm.String",
    "searchable": true
  },
  {
    "name": "content_vector",
    "type": "Collection(Edm.Single)",
    "searchable": true,
    "vector_search_configuration": "default"
  },
  {
    "name": "metadata",
    "type": "Edm.String",
    "searchable": true
  },
  {
    "name": "source",
    "type": "Edm.String",
    "filterable": true
  }
]
```
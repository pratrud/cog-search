# Extended Json Html Loader/Splitter

## Table of Contents <!-- omit in toc -->

- [Overview](#overview)
- [Implementation of Json HTML Loader](#implementation-of-json-html-loader)
- [Implementation of Json HTML Splitter](#implementation-of-json-html-splitter)

## Overview

Regular splitters in LangChain and custom loader/splitter are simply unwrapping/skipping html tags which causes to not use html tags to split the large documents. In order to overcome to this problem, we intend to develop an extended Json Html Loader to read the documents with their html tags and process the html content of documents and use them to split the documents based on specific HTML tags such as header tags or paragraph tags with specific font size.

## Implementation of Json HTML Loader

Note that the extended version of JSON Html loader will be enabled if `remove_all_tags` set to `false`. The extended JSON Html Loader will be implemented in the following steps:

1. Read the documents with their html tags from a specified property from a Json file.
1. Some of the care HTML pages does not follow the modern HTML techniques to layout the sections and headers. For example, in some cases, instead of `div` and `header` tags, they are using `span` or `p` tags with hard coded font size. In order to handle such cases properly and not loose HTML sections, we will convert all of such tags to `header` tags which will be consumed by a new splitter. In order to make this process flexible, we will use an XSLT file to transfer the content of the raw html files into a new HTML content with a desired format with `header` tags. This `XSLT` file will find all nodes in the HTML content with a specific font size and convert them to a `header` tag as below: 

    ```xml
    <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
      <!-- Copy all nodes and attributes by default -->
      <xsl:template match="@*|node()">
        <xsl:copy>
          <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
      </xsl:template>
     
      <!-- Match any element that has a font-size attribute larger than 20px -->
      <xsl:template match="*[@style[contains(., 'font-size')]]">
        <!-- Extract the font size value from the style attribute -->
        <xsl:variable name="font-size" select="substring-before(substring-after(@style, 'font-size:'), 'px')" />
        <!-- Check if the font size is larger than 20 -->
        <xsl:choose>
          <xsl:when test="$font-size > 20">
            <!-- Replace the element with a header tag -->
            <h1>
              <xsl:apply-templates select="@*|node()"/>
            </h1>
          </xsl:when>
          <xsl:otherwise>
            <!-- Keep the original element -->
            <xsl:copy>
              <xsl:apply-templates select="@*|node()"/>
            </xsl:copy>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:template>
    </xsl:stylesheet>
    ```

1. In the next step, the loader will decompose all unwanted tags that has been specified using `html_tags_to_decompose` property in the configuration file. For example, if the `html_tags_to_decompose` property is set to `["script", "style"]`, the loader will remove all `script` and `style` tags from the HTML content of the documents.
1. Remove all tags for which the content is empty. We noticed that in the care HTML files, there are several tags with empty content which are not useful for the document processing. For example, the following tag is not useful for the document processing and will be removed from the HTML content of the document:

    ```html
    <div class="col-md-12 col-sm-12 col-xs-12">
    </div>
    ```

1. Processing tables and converting them to a proper `CSV` format. This step will use `panda` package to convert a table to `CSV`. The library also handles the tables with nested tables. Since in most documents, the tables have been also used to layout the page, we will unwrap all tables that have less than 2 rows otherwise the tables will be replaced with a `CSV` tag and the content will be a comma separated values of the table cells. 
1. By default all attributes in all types of tags will be removed except those attributes that have been listed using `html_attrs_to_keep` property in the configuration file. For example, if the `html_attrs_to_keep` property is set to `["colspan", "rowspan"]`, the loader will keep only `colspan` and `rowspan` attributes in all types of tags. These two attributes are mainly used in `table` tags to specify the number of columns and rows in the table which will be used in the document processing when converting the HTML table content to the plain text.
1. The next step will be removing all unwanted special characters including `"\u00a0"`, `"¶"`

## Implementation of Json HTML Splitter

The extended Json HTML splitter will be implemented in the following steps:

1. Find all tags specified by `headers_to_split_on`. For example, if the `headers_to_split_on` property is set to `[["h1", "Header 1"]]`, the splitter will find all `body` and `h1` tags in the HTML content of the document. The reason we are considering `body` tag is that in some cases, the document does not have any `h1` tag or the page content not starting with a `h1` tag. In such cases, we will consider the first section starting from `body` tag and ending to a `h1` tag. The next section will start from the first `h1` tag and ends to the next `h1` tag. The last section will start from the last `h1` tag and ends to the end of the document.
1. Based on the metadata name specified using `headers_to_split_on`, the splitter will create a new section for each header tag. For example, if the `headers_to_split_on` property is set to `[["h1", "Header 1"]]`, the splitter will create a new section for each `h1` tag and the section name will be the inner HTML content of the tag and a new metadata named `Header 1` with the inner HTML content of the tag will be added to the section metadata.
1. At the end, we will use `CharacterTextSplitter` as a default splitter to split the sections into multiple documents with the same metadata as the original section if the section is larger than specified chunk size. Otherwise, the section will be kept as a single document. All of the `CharacterTextSplitter` constructor parameters can also be used as a parameter for the new splitter including `chunk_size`, `chunk_overlap` and `separator`. Note than the default splitter will also handle empty sections and will not create a new document for them.

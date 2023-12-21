"""Loader that loads data from JSON."""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from unstructured.documents.html import HTMLDocument
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
import bs4
import os
import pandas
from io import StringIO
import pathlib
import copy
use_jq = os.environ.get("USE_JQ", "true").lower() == "true"


class JSONLoaderWithHtml(BaseLoader):
    """Loads a JSON file and references a jq schema provided to load the text into
    documents.

    Example:
        [{"text": ...}, {"text": ...}, {"text": ...}] -> schema = .[].text
        {"key": [{"text": ...}, {"text": ...}, {"text": ...}]} -> schema = .key[].text
        ["", "", ""] -> schema = .[]
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        jq_schema: str,
        content_key: Optional[str] = None,
        metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
        text_content: bool = True,
        remove_all_tags : Optional[bool] = True,
        html_tags_to_decompose: Optional[list[str]] = ["style", "script","img"],
        html_tags_to_unwrap: Optional[list[str]] =["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "a", "span", "div"],
        html_attrs_to_keep: Optional[list[str]] = ["colspan", "rowspan"],
    ):
        """Initialize the JSONLoader.

        Args:
            file_path (Union[str, Path]): The path to the JSON file.
            jq_schema (str): The jq schema to use to extract the data or text from
                the JSON.
            content_key (str): The key to use to extract the content from the JSON if
                the jq_schema results to a list of objects (dict).
            metadata_func (Callable[Dict, Dict]): A function that takes in the JSON
                object extracted by the jq_schema and the default metadata and returns
                a dict of the updated metadata.
            text_content (bool): Boolean flag to indicates whether the content is in
                string format, default to True
            remove_all_tags (bool): a boolean flag to remove all html tags or not
            html_tags_to_decompose (Optional[list[str]]): a list of tags that should be removed with their content
            html_tags_to_unwrap (Optional[list[str]]): a list of the tags to unwrap. These flags will be removed but their content will be remained.
            html_attrs_to_keep (Optional[list[str]]): a list of attributes to be retained.
        """
        try:
            if use_jq:
                import jq  # noqa:F401
                self._jq_schema = jq.compile(jq_schema)
        except ImportError:
            raise ImportError(
                "jq package not found, please install it with `pip install jq`"
            )

        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self._metadata_func = metadata_func
        self._text_content = text_content
        self._remove_all_tags  = remove_all_tags 
        self._html_tags_to_decompose = html_tags_to_decompose
        self._html_tags_to_unwrap = html_tags_to_unwrap
        self._html_attrs_to_keep = html_attrs_to_keep
        # self._html_unwanted_strings = html_unwanted_strings

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""
        try:
            if use_jq:
                data = self._jq_schema.input(json.loads(self.file_path.read_text()))
            else:
                data = [json.loads(self.file_path.read_text())]
        except:
            if use_jq:
                data = self._jq_schema.input(json.loads(self.file_path.read_text(encoding='utf-8-sig')))
            else:
                data = [json.loads(self.file_path.read_text(encoding='utf-8-sig'))]

        # Perform some validation
        # This is not a perfect validation, but it should catch most cases
        # and prevent the user from getting a cryptic error later on.
        if self._content_key is not None:
            self._validate_content_key(data)

        docs = []
        for i, sample in enumerate(data, 1):
            metadata = dict(
                source=str(self.file_path),
                seq_num=i,
            )
            text = self._get_text(sample=sample, metadata=metadata)
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
    
    def convert_possible_tags_to_header(self, html_content: str)->str:
        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(
                "Unable to import lxml, please install with `pip install lxml`."
            ) from e
        # use lxml library to parse html document and return xml ElementTree
        parser = etree.HTMLParser()
        tree = etree.parse(StringIO(html_content), parser)

        # document transformation for "structure-aware" chunking is handled with xsl.
        # this is needed for htmls files that not using div tags to layout pages
        xslt_path = (
            pathlib.Path(__file__).parent
            / "converting_to_header.xslt"
        )
        xslt_tree = etree.parse(xslt_path)
        transform = etree.XSLT(xslt_tree)
        result = transform(tree)
        return str(result)

    def process_tags(self, html_content: str):

        html_content = self.convert_possible_tags_to_header(html_content)

        soup = bs4.BeautifulSoup(html_content, "html.parser")

        for tag in self._html_tags_to_decompose:
            for match in soup.findAll(tag):
                match.decompose()

        for tag in self._html_tags_to_unwrap:
            for match in soup.findAll(tag):
                match.unwrap()

        # remove all tags with empty content
        for match in soup.findAll():
            if match.text.strip() == "":
                match.decompose()

        for match in soup.findAll():
            match.attrs = {key: value for key, value in match.attrs.items() if key in self._html_attrs_to_keep}

        processed_content = self.process_tables(soup)

        # removing all unwanted strings
        for string_to_remove in ["\u00a0", "¶"]:
            processed_content = processed_content.replace(string_to_remove, " ")

        # removing multi spaces
        processed_content = ' '.join(processed_content.split(' '))

        # removing multi new lines
        processed_content = '\n'.join([s for s in processed_content.split('\n') if s.strip() != ''])

        return processed_content

    def process_tables(self, soup: bs4.BeautifulSoup):
        tables = soup.find_all('table')
        table_len = []

        for table in tables:
            trs =  table.find_all("tr")
            ths = table.find_all("th")
            if len(trs)+len(ths)  > 1:
                df = pandas.read_html(StringIO(str(table)), header=0)[0]

                csv_string = df.to_csv(index=False)

                # Replace table with new element that contains CSV string
                new_element = soup.new_tag('csv') # create a new element with <pre> tag
                new_element.string = csv_string # set its text to CSV string

                if table.parent != None:
                    table.insert_after(new_element)
                    table.decompose()
                else:
                    table.replace_with(new_element)
            else:
                div = soup.new_tag("div")
                for tag_name in ["tbody", "tr", "td", "th"]:
                    for tag in table.find_all(tag_name):
                        tag.unwrap()

                for content in copy.copy(table.contents):
                    div.append(content)

                if div.contents != None and len(div.contents) != 0:
                    if table.parent == None:
                        table.replace_with(div)
                    else:
                        table.insert_after(div)
                        table.decompose()
                else:
                    table.decompose()


        file_content = str(soup)
        return file_content
    
    def _get_text(self, sample: Any, metadata: dict) -> str:
        """Convert sample to string format"""
        if self._content_key is not None:
            raw_content = sample.get(self._content_key)
            if self._remove_all_tags :
                content = HTMLDocument.from_string(raw_content)  
            else:
                content = self.process_tags(raw_content)

            if self._metadata_func is not None:
                # We pass in the metadata dict to the metadata_func
                # so that the user can customize the default metadata
                # based on the content of the JSON object.
                metadata = self._metadata_func(sample, metadata)
        else:
            content = HTMLDocument.from_string(sample)

        if self._text_content and not isinstance(content, str):
            raise ValueError(
                f"Expected page_content is string, got {type(content)} instead. \
                    Set `text_content=False` if the desired input for \
                    `page_content` is not a string"
            )

        # In case the text is None, set it to an empty string
        elif isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content) if content else ""
        else:
            return str(content) if content is not None else ""

    def _validate_content_key(self, data: Any) -> None:
        """Check if content key is valid"""
        if use_jq:
            sample = data.first()
        else:
            sample = data[0]
            
        if not isinstance(sample, dict):
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict), \
                    so sample must be a dict but got `{type(sample)}`"
            )

        if sample.get(self._content_key) is None:
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict) \
                    with the key `{self._content_key}`"
            )

        if self._metadata_func is not None:
            sample_metadata = self._metadata_func(sample, {})
            if not isinstance(sample_metadata, dict):
                raise ValueError(
                    f"Expected the metadata_func to return a dict but got \
                        `{type(sample_metadata)}`"
                )

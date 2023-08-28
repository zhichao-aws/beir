import itertools
import textwrap
from typing import Type, List, Dict, Union, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection


class OpenSearchDataIngestor:

    def __init__(self, endpoint: str, port: str, timeout: int = 30, language: str = "english"):
        self.opensearch = OpenSearch(
            hosts=[{
                'host': endpoint,
                'port': port
            }],
            use_ssl=False,
            verify_certs=False,
            connection_class=RequestsHttpConnection,
            timeout=timeout
        )
        self.bulk_size = 200
        self.max_tokens = 512
        self.language = language

    def ingest(self, corpus: Dict[str, Dict[str, str]], index: str):
        for i in range(0, len(corpus), self.bulk_size):
            key_list = itertools.islice(corpus.keys(), i, i + self.bulk_size)

            def get_doc_text(full_string: str):
                str_as_list = textwrap.wrap(full_string, self.max_tokens, break_long_words=False,
                                            break_on_hyphens=False)
                return full_string if len(str_as_list) == 0 else str_as_list[0]
                # return ' '.join(full_string.split()[:500])

            actions = []
            _ = [
                actions.extend(
                    [{'index': {'_index': index, '_id': key_id}},
                     {'passage_text': get_doc_text((corpus[key_id]["title"] + ' ' + corpus[key_id]["text"]).strip()), 'text_key': corpus[key_id]['text'], 'title_key': corpus[key_id]['title']}])
                for key_id in key_list
            ]
            # actions[1::2] = [{'passage_text': corpus[key_id]['text']} for key_id in key_list]
            self.opensearch.bulk(
                index=index,
                body=actions)

            if i % 1000 == 0:
                print("Ingested " + str(i) + " documents")

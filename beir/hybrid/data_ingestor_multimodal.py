import itertools
import base64
import requests
from typing import Type, List, Dict, Union, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection


class OpenSearchDataIngestor_Flickr30:

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
        self.bulk_size = 20
        self.max_tokens = 512
        self.language = language

    def ingest(self, corpus: Dict[str, Dict[str, str]], index: str, corpus_method, shift_corpus):

        for i in range(0, len(corpus), self.bulk_size):

            if i < shift_corpus:
                continue

            key_list = itertools.islice(corpus.keys(), i, i + self.bulk_size)

            def get_base64_from_image(image_url: str):
                return base64.b64encode(requests.get(image_url).content).decode('utf-8')
            def get_content(corpus_doc):
                if corpus_method == 'image_text':
                    return {'image_id': corpus_doc["image_id"], 'image': get_base64_from_image(corpus_doc["image_url"]), 'caption_id': corpus_doc["caption_id"], 'caption': corpus_doc["caption"]}
                else:
                    return {'image_id': corpus_doc["image_id"], 'caption_id': corpus_doc["caption_id"], 'caption': corpus_doc["caption"]}

            actions = []
            _ = [
                actions.extend(
                    [{'index': {'_index': index, '_id': key_id}},
                     get_content(corpus[key_id])])
                for key_id in key_list
            ]
            # actions[1::2] = [{'passage_text': corpus[key_id]['text']} for key_id in key_list]
            self.opensearch.bulk(
                index=index,
                body=actions,
                refresh='true')

            if i % 1000 == 0:
                print("Ingested " + str(i) + " documents")

# from beir.retrieval.search.util import cos_sim, dot_score
import logging
import textwrap
import random
from typing import Dict, List
from opensearchpy import OpenSearch, RequestsHttpConnection
import base64
import requests

logger = logging.getLogger(__name__)


# Parent class for any dense model
class RetrievalOpenSearch_Flickr30k_Local:

    def __init__(self, endpoint: str, port: str, index_name: str, model_id: str,  static_content_folder, batch_size: int = 128,
                 corpus_chunk_size: int = 50000, timeout: int = 30, search_method: str = 'image_text',
                 pipeline_name: str = 'norm-pipeline', **kwargs):
        # model is class that provides encode_corpus() and encode_queries()
        self.took_time = {}
        self.batch_size = batch_size
        # self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        # self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True  # TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.results = {}
        self.index_name = index_name
        self.model_id = model_id
        self.search_method = search_method
        self.max_tokens = 512
        self.static_content_folder = static_content_folder

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
        self.pipeline_name = pipeline_name

    """
        Function does bm25 search only
    """
    def search_image_text(self,
                    corpus: Dict[str, Dict[str, str]],
                    queries: Dict[str, str],
                    top_k: int,
                    result_size: int,
                    return_sorted: bool = False, **kwargs) -> Dict[str, Dict[str, float]]:

        def get_base64_from_image(image_name: str):
            base_url = self.static_content_folder
            image_url = base_url + image_name
            image_file = open(image_url, "rb")
            return base64.b64encode(image_file.read()).decode('utf-8')

        def get_body_image_text(query_text, q_image_name):
            query_image = get_base64_from_image(q_image_name)
            return {
                'size': result_size,
                'query': {
                    'neural': {
                        'vector_embedding': {
                            'query_text': query_text,
                            'query_image': query_image,
                            'model_id': model_id,
                            'k': top_k
                        }
                    }
                }
            }

        def get_body_image(query_text, q_image_name):
            query_image = get_base64_from_image(q_image_name)
            return {
                'size': result_size,
                'query': {
                    'neural': {
                        'vector_embedding': {
                            'query_image': query_image,
                            'model_id': model_id,
                            'k': top_k
                        }
                    }
                }
            }

        def get_body_text(query_text, q_image_name):
            return {
                'size': result_size,
                'query': {
                    'neural': {
                        'vector_embedding': {
                            'query_text': query_text,
                            'model_id': model_id,
                            'k': top_k
                        }
                    }
                }
            }

        def get_body(k, query_text, q_image_name):
            searches = {
                'image_text': get_body_image_text,
                'image': get_body_image,
                'text': get_body_text
            }
            return searches[self.search_method](query_text, q_image_name)

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]

        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
                            reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")

        model_id = self.model_id
        index_name = self.index_name

        query_responses = []

        for q_id in range(0, len(query_ids)):
            q_text = queries[q_id]["caption"]
            q_image_name = queries[q_id]["image"]

            query_response = self.opensearch.search(index=index_name,
                                                    body=get_body(top_k, q_text, q_image_name))

            query_responses.append(query_response)
            if q_id % 50 == 0:
                print("Executed queries: " + str(q_id))

        for i in range(0, len(query_responses)):
            query_id = query_ids[i]
            for hit in query_responses[i]['hits']['hits']:
                corp_id = hit['_id']
                if corp_id != query_id:
                    self.results[query_id][corp_id] = hit['_score']

        return self.results

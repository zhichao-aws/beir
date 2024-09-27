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
class RetrievalOpenSearch_Flickr30k:

    def __init__(self, endpoint: str, port: str, index_name: str, model_id: str, batch_size: int = 128,
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

        def get_body_image_text(query_text, query_image):
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

        def get_body_image(query_text, query_image):
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

        def get_body_text(query_text, query_image):
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

        def get_body(k, query_text, query_image):
            searches = {
                'image_text': get_body_image_text,
                'image': get_body_image,
                'text': get_body_text
            }
            return searches[self.search_method](query_text, query_image)

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

        def get_doc_text(full_string: str):
            str_as_list = textwrap.wrap(full_string, self.max_tokens, break_long_words=False,
                                        break_on_hyphens=False)
            return full_string if len(str_as_list) == 0 else str_as_list[0]

        def get_base64_from_image(image_url: str):
            return base64.b64encode(requests.get(image_url).content).decode('utf-8')

        for q_id in range(0, len(query_ids)):
            q_text = queries[q_id]["caption"]
            q_image_url = queries[q_id]["coco_url"]
            q_image = get_base64_from_image(q_image_url)

            query_response = self.opensearch.search(index=index_name,
                                                    body=get_body(top_k, q_text, q_image))

            query_responses.append(query_response)
            if q_id % 50 == 0:
                print("Executed queries: " + str(q_id))


        ids = [[hit['_id']
                for hit in query_response['hits']['hits']]
               for query_response in query_responses]

        for i in range(0, len(query_responses)):
            query_id = query_ids[i]
            for hit in query_responses[i]['hits']['hits']:
                corp_id = hit['_id']
                if corp_id != query_id:
                    self.results[query_id][corp_id] = hit['_score']

        return self.results

    """
        Function works with vector based searches, knn/neural and hybrid
    """
    def search_vector(self,
                      corpus: Dict[str, Dict[str, str]],
                      queries: Dict[str, str],
                      top_k: int,
                      result_size: int,
                      return_sorted: bool = False, **kwargs) -> Dict[str, Dict[str, float]]:
        # def get_body(k, query_text, model_id):
        #     return {
        #         'size': top_k,
        #         'query': {
        #             'neural': {
        #                 'passage_embedding': {
        #                     'query_text': query_text,
        #                     'model_id': model_id,
        #                     'k': k
        #                 }
        #             }
        #         }
        #     }
        def get_body_hybrid(query_text):
            return {
                'size': result_size,
                'query': {
                    "hybrid": {
                        "queries": [
                            {
                                'neural': {
                                    'passage_embedding': {
                                        'query_text': query_text,
                                        'model_id': model_id,
                                        'k': top_k
                                    }
                                }
                            },
                            # {
                            #    'multi_match': {
                            #        'query': query_text,
                            #        'type': 'best_fields',
                            #        'fields': ['text_key', 'title_key'],
                            #        "tie_breaker": 0.5
                            #    }
                            # }
                            # {
                            #     'match': {
                            #         'title_key': {
                            #             'query': query_text
                            #         }
                            #     }
                            # },
                            {
                                'match': {
                                    'text_key': {
                                        'query': query_text
                                    }
                                }
                            }
                        ]
                    }
                }
            }

        def get_body_bool(query_text):
            return {
                'size': result_size,
                'query': {
                    "bool": {
                        "must": [
                            {
                                'neural': {
                                    'passage_embedding': {
                                        'query_text': query_text,
                                        'model_id': model_id,
                                        'k': top_k
                                    }
                                }
                            },
                            # {
                            #    'multi_match': {
                            #        'query': query_text,
                            #        'type': 'best_fields',
                            #        'fields': ['text_key', 'title_key'],
                            #        "tie_breaker": 0.5
                            #    }
                            # }
                            # {
                            #     'match': {
                            #         'title_key': {
                            #             'query': query_text
                            #         }
                            #     }
                            # },
                            {
                                'match': {
                                    'text_key': {
                                        'query': query_text
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        def get_body_neural(query_text):
            return {
                'size': result_size,
                'query': {
                    'neural': {
                        'passage_embedding': {
                            'query_text': query_text,
                            'model_id': model_id,
                            'k': top_k
                        }
                    }
                }
            }

        def get_body_vector(query_text):
            searches = {
                'neural': get_body_neural,
                'hybrid': get_body_hybrid,
                'bool': get_body_bool
            }
            return searches[self.search_method](query_text)

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        self.took_time = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]

        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
                            reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        # logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        model_id = self.model_id
        index_name = self.index_name

        query_responses = []

        def get_doc_text(full_string: str):
            str_as_list = textwrap.wrap(full_string, self.max_tokens, break_long_words=False,
                                        break_on_hyphens=False)
            return full_string if len(str_as_list) == 0 else str_as_list[0]

        logger.info("Starting warmup queries")
        for r in range(0, min(100, len(query_ids))):
            q = random.choice(queries)
            self.opensearch.search(index=index_name,
                                   body=get_body_vector(get_doc_text(q)),
                                   params={"search_pipeline": self.pipeline_name})
        logger.info("Finished warmup queries")

        for i in range(0, len(query_ids)):
            q = queries[i]

            query_response = self.opensearch.search(index=index_name,
                                                    body=get_body_vector(get_doc_text(q)),
                                                    params={"search_pipeline": self.pipeline_name})

            query_responses.append(query_response)
            if i % 50 == 0:
                print("Executed queries: " + str(i))

        ids = [[hit['_id']
                for hit in query_response['hits']['hits']]
               for query_response in query_responses]

        for i in range(0, len(query_responses)):
            query_id = query_ids[i]
            for hit in query_responses[i]['hits']['hits']:
                corp_id = hit['_id']
                if corp_id != query_id:
                    self.results[query_id][corp_id] = hit['_score']
            self.took_time[query_id] = int(query_responses[i]['took'])

        return self.results, self.took_time

# from beir.retrieval.search.util import cos_sim, dot_score
import logging
import textwrap
import random
from typing import Dict, List
from opensearchpy import OpenSearch, RequestsHttpConnection

logger = logging.getLogger(__name__)


# Parent class for any dense model
class RetrievalOpenSearch:

    def __init__(self, endpoint: str, port: str, index_name: str, model_id: str, batch_size: int = 128,
                 corpus_chunk_size: int = 50000, timeout: int = 30, search_method: str = 'bm25',
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
    def search_bm25(self,
                    corpus: Dict[str, Dict[str, str]],
                    queries: Dict[str, str],
                    top_k: int,
                    return_sorted: bool = False, **kwargs) -> Dict[str, Dict[str, float]]:

        def get_body_bm25(query_text):
            return {
                'size': top_k,
                'query': {
                    'multi_match': {
                        'query': query_text,
                        'type': 'best_fields',
                        'fields': ['text_key', 'title_key'],
                        "tie_breaker": 0.5
                    }
                    # 'match': {
                    #     'text_key': {
                    #         'query': query_text
                    #     }
                    # }
                }
            }

        def get_body(k, query_text):
            searches = {
                'bm25': get_body_bm25
            }
            return searches[self.search_method](query_text)

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

        for i in range(0, len(query_ids)):
            query_id = query_ids[i]
            q = queries[i]

            query_response = self.opensearch.search(index=index_name,
                                                    body=get_body(top_k, get_doc_text(q)))

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

        return self.results

    # def search_bm25(self,
    #                 corpus: Dict[str, Dict[str, str]],
    #                 queries: Dict[str, str],
    #                 top_k: int,
    #                 return_sorted: bool = False, **kwargs) -> Dict[str, Dict[str, float]]:
    #     # def get_body(k, query_text, model_id):
    #     #     return {
    #     #         'size': top_k,
    #     #         'query': {
    #     #             'neural': {
    #     #                 'passage_embedding': {
    #     #                     'query_text': query_text,
    #     #                     'model_id': model_id,
    #     #                     'k': k
    #     #                 }
    #     #             }
    #     #         }
    #     #     }
    #
    #     def get_body_bm25(k, query_text):
    #         return {
    #             'size': top_k,
    #             'query': {
    #                 # 'multi_match': {
    #                 #     'query': query_text,
    #                 #     'type': 'best_fields',
    #                 #     'fields': ['text_key', 'title_key'],
    #                 #     "tie_breaker": 0.5
    #                 # }
    #                 'match': {
    #                     'text_key': {
    #                         'query': query_text
    #                     }
    #                 }
    #             }
    #         }
    #
    #     def get_body(k, query_text):
    #         searches = {
    #             'bm25': get_body_bm25
    #         }
    #         return searches[self.search_method](k, query_text)
    #
    #     # def get_body(k, query_text, model_id):
    #     #     return {
    #     #         'size': top_k,
    #     #         'query': {
    #     #             "hybrid": {
    #     #                 "queries": [
    #     #                     {
    #     #                         'neural': {
    #     #                             'passage_embedding': {
    #     #                                 'query_text': query_text,
    #     #                                 'model_id': model_id,
    #     #                                 'k': k
    #     #                             }
    #     #                         }
    #     #                     },
    #     #                     {
    #     #                         'term': {
    #     #                              'passage_text': query_text
    #     #                         }
    #     #                     }
    #     #                 ]
    #     #             }
    #     #         }
    #     #     }
    #     # Create embeddings for all queries using model.encode_queries()
    #     # Runs semantic search against the corpus embeddings
    #     # Returns a ranked list with the corpus ids
    #     # if score_function not in self.score_functions:
    #     #    raise ValueError(
    #     #        "score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(
    #     #            score_function))
    #
    #     logger.info("Encoding Queries...")
    #     query_ids = list(queries.keys())
    #     self.results = {qid: {} for qid in query_ids}
    #     queries = [queries[qid] for qid in queries]
    #     # query_embeddings = self.model.encode_queries(
    #     #    queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar,
    #     #    convert_to_tensor=self.convert_to_tensor)
    #
    #     logger.info("Sorting Corpus by document length (Longest first)...")
    #
    #     corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
    #                         reverse=True)
    #     corpus = [corpus[cid] for cid in corpus_ids]
    #
    #     logger.info("Encoding Corpus in batches... Warning: This might take a while!")
    #     # logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
    #
    #     model_id = self.model_id
    #     index_name = self.index_name
    #
    #     query_responses = []

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
                            {
                                'match': {
                                    'title_key': {
                                        'query': query_text
                                    }
                                }
                            },
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
                            {
                                'match': {
                                    'title_key': {
                                        'query': query_text
                                    }
                                }
                            },
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

            search_params = {}
            if self.search_method == 'hybrid':
                search_params["search_pipeline"] = self.pipeline_name
            query_response = self.opensearch.search(index=index_name,
                                                    body=get_body_vector(get_doc_text(q)),
                                                    params=search_params)

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

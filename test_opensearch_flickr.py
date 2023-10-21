from beir import util, LoggingHandler
from beir.hybrid.evaluation import EvaluateRetrieval
from beir.hybrid.search_amazon import RetrievalOpenSearch_Amazon
from beir.hybrid.data_ingestor_flickr import OpenSearchDataIngestor_Flickr
import csv

import logging
import pathlib, os, getopt, sys


def main(argv):
    opts, args = getopt.getopt(argv, "h:p:i:m:n:o:",
                               ["os_host=", "os_port=", "os_index=", "os_model_id=", "num_of_runs=", "operation="])
    dataset = 'amazon'
    endpoint = ''
    port = ''
    index = ''
    model_id = ''
    num_of_runs = 2
    operation = "evaluate"
    for opt, arg in opts:
        if opt in ("-h", "-os_host"):
            endpoint = arg
        elif opt in ("-p", "-os_port"):
            port = arg
        elif opt in ("-i", "-os_index"):
            index = arg
        elif opt in ("-m", "-os_model_id"):
            model_id = arg
        elif opt in ("-n", "-num_of_runs"):
            num_of_runs = int(arg)
        elif opt in ("-o", "-operation"):
            operation = arg


    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    print("Preparing corpus")
    corpus = {}

    with open('/Users/gaievski/dev/opensearch/datasets/flickr30k_images/results.csv', newline='') as csvfile:
        data_reader = csv.reader(csvfile, delimiter='|', quotechar='|')
        for row in data_reader:
            doc_id = row[0].split('.')[0]
            if doc_id not in corpus:
                corpus[doc_id] = {"text": row[2].strip(), "id": row[0].strip()}
                print(corpus[doc_id])

    if operation == 'ingest' or operation == 'both':
        ingest_data(corpus, endpoint, index, port)


def ingest_data(corpus, endpoint, index, port):
    OpenSearchDataIngestor_Flickr(endpoint, port).ingest(corpus, index=index)


def evaluate(corpus, endpoint, index, model_id, port, qrels, queries, num_of_runs):
    # This k values are being used for BM25 search
    # bm25_k_values = [1, 3, 5, 10, 100, min(9999, len(corpus))]
    bm25_k_values = [1, 3, 5, 10, 100]
    # This K values are being used for dense model search
    model_k_values = [1, 3, 5, 10, 100]
    # this k values are being used for scoring
    k_values = [5, 10, 100]
    # for method in ['bm25', 'neural', 'hybrid']:
    # for method in ['hybrid']:
    # for method in ['neural']:

    # for method in ['bm25']:
    #     print('starting search method ' + method)
    #     os_retrival = RetrievalOpenSearch(endpoint, port,
    #                                       index_name=index,
    #                                       model_id=model_id,
    #                                       search_method=method,
    #                                       pipeline_name='norm-pipeline')
    #     retriever = EvaluateRetrieval(os_retrival, bm25_k_values)
    #     result_size = max(bm25_k_values)
    #     results = os_retrival.search_bm25(corpus, queries, top_k=result_size)
    #     ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
    #     print('--- end of results for ' + method)

    #for method in ['neural', 'hybrid']:
    for method in ['hybrid']:
    # for method in ['hybrid', 'bool']:
        print('starting search method ' + method)
        os_retrival = RetrievalOpenSearch_Amazon(endpoint, port,
                                          index_name=index,
                                          model_id=model_id,
                                          search_method=method,
                                          pipeline_name='norm-pipeline')
        retriever = EvaluateRetrieval(os_retrival, model_k_values)  # or "cos_sim" for cosine similarity
        top_k = max(model_k_values)
        result_size = max(bm25_k_values)
        # results = retriever.retrieve(corpus, queries)
        all_experiments_took_time = []
        for run in range(0, num_of_runs) :
            results, took_time = os_retrival.search_vector(corpus, queries, top_k=top_k, result_size=result_size)
            all_experiments_took_time.append(took_time)
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
        # print('Total time: ' + str(total_time))
        retriever.evaluate_time(all_experiments_took_time)
        print('--- end of results for ' + method)


if __name__ == "__main__":
    main(sys.argv[1:])
from beir import util, LoggingHandler
from beir.datasets.data_loader_coco_2014 import DataLoader
from beir.hybrid.evaluation import EvaluateRetrieval
from beir.hybrid.search_flickr30k import RetrievalOpenSearch_Flickr30k
from beir.hybrid.data_ingestor_multimodal import OpenSearchDataIngestor_Flickr30

import logging
import pathlib, os, getopt, sys


def main(argv):
    opts, args = getopt.getopt(argv, "d:u:h:p:i:m:n:o:l:e:q:c:s:",
                               ["dataset=", "dataset_url=", "os_host=", "os_port=", "os_index=", "os_model_id=",
                                "num_of_runs=", "operation=", "pipelines=", "method=", "nqueries=", "corpus_method=", "shiftcorpus="])
    dataset = 'captions_val2014.json'
    url = ''
    endpoint = ''
    port = ''
    index = ''
    model_id = ''
    num_of_runs = 2
    operation = "ingest"
    pipelines = 'norm-pipeline'
    mmethod = 'image_text'
    num_of_queries = 200
    corpus_method = 'image_text'
    shift_corpus = 0
    for opt, arg in opts:
        if opt in ("-d", "-dataset"):
            dataset = arg
        elif opt in ("-u", "-dataset_url"):
            url = arg
        elif opt in ("-h", "-os_host"):
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
        elif opt in ("-l", "-pipelines"):
            pipelines = arg
        elif opt in ("-e", "-method"):
            mmethod = arg
        elif opt in ("-q", "-nqueries"):
            num_of_queries = int(arg)
        elif opt in ("-c", "-corpus_method"):
            corpus_method = arg
        elif opt in ("-s", "-shiftcorpus"):
            shift_corpus = int(arg)

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #### /print debug information to stdout
    #### Download scifact.zip dataset and unzip the dataset
    # dataset = "nfcorpus"
    # dataset = "trec-covid"
    # dataset = "arguana"
    # dataset = 'fiqa'
    # url = url.format(dataset)
    # out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), dataset)
    data_path = pathlib.Path(url).absolute()

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = DataLoader(data_folder=data_path, corpus_file=dataset, num_of_queries=num_of_queries).load(corpus_method=corpus_method)

    if operation == 'ingest' or operation == 'both':
        ingest_data(corpus, endpoint, index, port, corpus_method, shift_corpus)

    if operation == 'evaluate' or operation == 'both':
        evaluate(corpus, endpoint, index, model_id, port, qrels, queries, num_of_runs, pipelines, mmethod)


def ingest_data(corpus, endpoint, index, port, corpus_method="text", shift_corpus=0):
    OpenSearchDataIngestor_Flickr30(endpoint, port).ingest(corpus, index=index, corpus_method=corpus_method, shift_corpus=shift_corpus)


def evaluate(corpus, endpoint, index, model_id, port, qrels, queries, num_of_runs, pipelines, mmethod):
    # This k values are being used for BM25 search
    # bm25_k_values = [1, 3, 5, 10, 100, min(9999, len(corpus))]
    bm25_k_values = [100]
    # This K values are being used for dense model search
    model_k_values = [100]
    # this k values are being used for scoring
    k_values = [1, 3, 5, 10, 25, 100]

    mm = mmethod.split(',')

    for method in get_multimodal_methods(mm):
        print('starting search method ' + method)
        os_retrival = RetrievalOpenSearch_Flickr30k(endpoint, port,
                                          index_name=index,
                                          model_id=model_id,
                                          search_method=method,
                                          pipeline_name=pipelines.split(',')[0])
        retriever = EvaluateRetrieval(os_retrival, bm25_k_values)
        result_size = max(bm25_k_values)
        top_k = max(model_k_values)
        results = os_retrival.search_image_text(corpus, queries, top_k=top_k, result_size=result_size)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
        print('--- end of results for ' + method)

    # for method in ['neural', 'hybrid']:
    for method in get_vector_methods(mm):
        # for method in ['hybrid', 'bool']:
        for pipeline in pipelines.split(','):
            print('starting search method ' + method + " for pipeline " + pipeline)
            os_retrival = RetrievalOpenSearch_Flickr30k(endpoint, port,
                                              index_name=index,
                                              model_id=model_id,
                                              search_method=method,
                                              pipeline_name=pipeline)
            retriever = EvaluateRetrieval(os_retrival, model_k_values)  # or "cos_sim" for cosine similarity
            top_k = max(model_k_values)
            result_size = max(bm25_k_values)
            # results = retriever.retrieve(corpus, queries)
            all_experiments_took_time = []
            for run in range(0, num_of_runs):
                results, took_time = os_retrival.search_vector(corpus, queries, top_k=top_k, result_size=result_size)
                all_experiments_took_time.append(took_time)
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
            # print('Total time: ' + str(total_time))
            retriever.evaluate_time(all_experiments_took_time)
            print('--- end of results for ' + method + " and pipeline " + pipeline)

    # method = 'hybrid'
    # print('starting search method ' + method)
    # os_retrival = RetrievalOpenSearch(endpoint, port,
    # index_name=index,
    # model_id=model_id,
    # search_method=method,
    # pipeline_name='norm-pipeline')
    # retriever = EvaluateRetrieval(os_retrival, bm25_k_values, model_k_values)  # or "cos_sim" for cosine similarity
    # results = retriever.retrieve(corpus, queries)
    # results = retriever.search(corpus, queries, top_k)
    # ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
    # print('--- end of results for ' + method)


def get_vector_methods(mm):
    vector_methods = []
    if 'neural' in mm:
        vector_methods.append('neural')
    if 'hybrid' in mm:
        vector_methods.append('hybrid')
    if 'bool' in mm:
        vector_methods.append('bool')
    return vector_methods

def get_multimodal_methods(mm):
    multimodal_methods = []
    if 'text' in mm:
        multimodal_methods.append('text')
    if 'image' in mm:
        multimodal_methods.append('image')
    if 'image_text' in mm:
        multimodal_methods.append('image_text')
    return multimodal_methods


if __name__ == "__main__":
    main(sys.argv[1:])

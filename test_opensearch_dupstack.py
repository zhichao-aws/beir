from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.hybrid.evaluation import EvaluateRetrieval
from beir.hybrid.search import RetrievalOpenSearch
from beir.hybrid.data_ingestor import OpenSearchDataIngestor

import logging
import pathlib, os, getopt, sys


def main(argv):
    opts, args = getopt.getopt(argv, "u:h:p:i:m:n:o:l:e:f:",
                               ["dataset_url=", "os_host=", "os_port=", "os_index=", "os_model_id=", "num_of_runs=", "operation=", "pipelines=", "method=", "subset="])
    dataset = 'cqadupstack'
    url = ''
    endpoint = ''
    port = ''
    index = ''
    model_id = ''
    num_of_runs = 2
    operation = "evaluate"
    pipelines = 'norm-pipeline'
    mmethod = 'hybrid'
    subset = ''
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
        elif opt in ("-f", "-subset"):
            subset = arg


    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #### /print debug information to stdout
    #### Download scifact.zip dataset and unzip the dataset
    url = url.format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    corpus = {}
    queries = {}
    qrels = {}
    mega_corpus = []
    mega_queries = []
    mega_qrels = []
    data_name = []
    total_len_corp = 0
    total_len_q = 0
    total_len_qrels = 0
    for filename in os.listdir(data_path):
        f = os.path.join(data_path, filename) + "/"
        print(f)
        corpus, queries, qrels = GenericDataLoader(data_folder=f).load(split="test")
        mega_corpus.append(corpus)
        mega_qrels.append(qrels)
        mega_queries.append(queries)
        data_name.append(filename)

    #### Provide the data_path where scifact has been downloaded and unzipped
    #corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    if operation == 'ingest' or operation == 'both':
        ingest_data(mega_corpus, data_name, endpoint, index, port, subset)

    if operation == 'evaluate' or operation == 'both':
        evaluate(mega_corpus, data_name, endpoint, index, model_id, port, qrels, queries, num_of_runs, pipelines, mmethod, subset)


def ingest_data(mega_corpus, data_name, endpoint, index, port, subset):
    for i in range(len(mega_corpus)):
        # print(i)
        if data_name[i] == subset:
            print("Experiment for ", data_name[i], " is running")
            OpenSearchDataIngestor(endpoint, port).ingest(mega_corpus[i], index=index)


def evaluate(mega_corpus, data_name, endpoint, index, model_id, port, qrels, queries, num_of_runs, pipelines, mmethod, subset):
    # This k values are being used for BM25 search
    # bm25_k_values = [1, 3, 5, 10, 100, min(9999, len(corpus))]
    bm25_k_values = [1, 3, 5, 10, 100]
    # This K values are being used for dense model search
    model_k_values = [1, 3, 5, 10, 100]
    # this k values are being used for scoring
    k_values = [5, 10, 100]

    mm = mmethod.split(',')

    for i in range(len(mega_corpus)):
        # print(i)
        if subset != data_name[i]:
            continue
        print("Experiment for ", data_name[i], " is running")
        corpus = mega_corpus[i]

        if 'bm25' in mm:
            method = 'bm25'
            print('starting search method ' + method)
            os_retrival = RetrievalOpenSearch(endpoint, port,
                                              index_name=index,
                                              model_id=model_id,
                                              search_method=method,
                                              pipeline_name=pipelines.split(',')[0])
            retriever = EvaluateRetrieval(os_retrival, bm25_k_values)
            result_size = max(bm25_k_values)
            results = os_retrival.search_bm25(corpus, queries, top_k=result_size)
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
            print('--- end of results for ' + method)

        #for method in ['neural', 'hybrid']:
        for method in get_vector_methods(mm):
            #for method in ['hybrid', 'bool']:
                for pipeline in pipelines.split(','):
                    print('starting search method ' + method + " for pipeline " + pipeline)
                    os_retrival = RetrievalOpenSearch(endpoint, port,
                                                      index_name=index,
                                                      model_id=model_id,
                                                      search_method=method,
                                                      pipeline_name=pipeline)
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
                    print('--- end of results for ' + method + " and pipeline " + pipeline)


def get_vector_methods(mm):
    vector_methods = []
    if 'neural' in mm:
        vector_methods.append('neural')
    if 'hybrid' in mm:
        vector_methods.append('hybrid')
    if 'bool' in mm:
        vector_methods.append('bool')
    return vector_methods


if __name__ == "__main__":
    main(sys.argv[1:])

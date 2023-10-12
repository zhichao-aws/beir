import sys
from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv

logger = logging.getLogger(__name__)

class DataLoader:

    def __init__(self, num_of_queries: int, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl",
                 qrels_folder: str = "qrels", qrels_file: str = ""):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

        self.num_of_queries = num_of_queries

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load(self, corpus_method="text") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        #self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        #self.check(fIn=self.corpus_file, ext="jsonl")
        #self.check(fIn=self.query_file, ext="jsonl")
        #self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus(corpus_method)
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if not len(self.qrels):
            self._load_qrels()
            #self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:

        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def _load_corpus(self, corpus_method='text'):

        with open(self.corpus_file, newline='') as csvfile:
            next(csvfile)
            csv.field_size_limit(sys.maxsize)
            data_reader = csv.reader(csvfile, delimiter='|', quotechar='|')
            for row in data_reader:
                image_name = row[0].strip()
                caption_idx = row[1].strip()
                if int(caption_idx) <= 3:
                    caption = row[2].strip()
                    doc_id = image_name + "_" + caption_idx
                    self.corpus[doc_id] = { "id": doc_id, "caption": caption, 'image_name': image_name }

    def _load_queries(self):
        with open(self.corpus_file, newline='') as csvfile:
            next(csvfile)
            csv.field_size_limit(sys.maxsize)
            data_reader = csv.reader(csvfile, delimiter='|', quotechar='|')
            for row in data_reader:
                image_name = row[0].strip()
                caption_idx = row[1].strip()
                if int(caption_idx) == 4:
                    caption = row[2].strip()
                    doc_id = image_name + "_" + caption_idx
                    self.queries[doc_id] = {
                        "caption": caption,
                        "image": image_name
                    }

        if len(self.queries) > self.num_of_queries:
            count = min(self.num_of_queries, len(self.queries.keys()))
            new_dict = {}
            c = 0
            for k,v in self.queries.items():
                new_dict[k] = v
                c = c + 1
                if c >= count:
                    break
            self.queries = new_dict

    def _load_qrels(self):

        with open(self.corpus_file, newline='') as csvfile:
            next(csvfile)
            csv.field_size_limit(sys.maxsize)
            data_reader = csv.reader(csvfile, delimiter='|', quotechar='|')
            for row in data_reader:
                image_name =row[0].strip()
                caption_idx = row[1].strip()
                if int(caption_idx) <= 3:
                    caption = row[2].strip()
                    doc_id = image_name + "_" + caption_idx
                    query_id = image_name + "_4"
                    query_id, corpus_id, score = query_id, doc_id, 1
                    if query_id not in self.qrels:
                        self.qrels[query_id] = {corpus_id: score}
                    else:
                        self.qrels[query_id][corpus_id] = score
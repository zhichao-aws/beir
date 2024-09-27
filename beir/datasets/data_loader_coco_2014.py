from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging

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

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

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

        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        img_url_by_image_id = {}
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                images = line.get("images")
                for img in images:
                    img_url_by_image_id[str(img["id"])] = img["coco_url"]

        caption_by_image_id = {}
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                captions = line.get("annotations")

                for caption in captions:
                    if caption["image_id"] not in caption_by_image_id.keys() or caption_by_image_id[caption["image_id"]] <= 2:
                        if corpus_method == 'image_text':
                            self.corpus[caption["id"]] = {
                                "caption": caption["caption"],
                                "image_id": caption["image_id"],
                                "caption_id": caption["id"],
                                "image_url": img_url_by_image_id[str(caption["image_id"])]
                            }
                        elif corpus_method == 'text':
                            self.corpus[caption["id"]] = {
                                "caption": caption["caption"],
                                "image_id": caption["image_id"],
                                "caption_id": caption["id"]
                            }

                        if caption["image_id"] not in caption_by_image_id.keys():
                            caption_by_image_id[caption["image_id"]] = 1

                        else:
                            caption_by_image_id[caption["image_id"]] = caption_by_image_id[caption["image_id"]] + 1
                    else:
                        #print("Drop caption with id " + str(caption["id"]))
                        continue

    def _load_queries(self):
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        caption_by_image_id = {}
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                captions = line.get("annotations")

                for caption in captions:
                    if caption["image_id"] not in caption_by_image_id.keys():
                        caption_by_image_id[caption["image_id"]] = 1
                    elif caption_by_image_id[caption["image_id"]] <= 2:
                        caption_by_image_id[caption["image_id"]] = caption_by_image_id[caption["image_id"]] + 1
                    else:
                        q_id = str(caption["image_id"])
                        self.queries[q_id] = {
                            "caption": caption["caption"],
                            "image_id": str(caption["image_id"]),
                            "caption_id": str(caption["id"])
                        }

        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                images = line.get("images")

                for img in images:
                    if str(img["id"]) not in self.queries.keys():
                        continue
                    self.queries[str(img["id"])]["coco_url"] = img["coco_url"]

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
        '''
        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")
        '''
    def _load_qrels(self):
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        caption_by_image_id = {}
        c = 0
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                captions = line.get("annotations")

                for caption in captions:
                    if caption["image_id"] not in caption_by_image_id.keys() or caption_by_image_id[caption["image_id"]] <= 2:
                        # self.qrels[caption["id"]] = {
                        #     "caption": caption["caption"],
                        #     "image_id": caption["image_id"],
                        #     "caption_id": caption["id"]
                        # }
                        query_id, corpus_id, score = str(caption["image_id"]), str(caption["id"]), 1
                        if query_id not in self.qrels:
                            self.qrels[query_id] = {corpus_id: score}
                        else:
                            self.qrels[query_id][corpus_id] = score

                        if caption["image_id"] not in caption_by_image_id.keys():
                            caption_by_image_id[caption["image_id"]] = 1

                        else:
                            caption_by_image_id[caption["image_id"]] = caption_by_image_id[caption["image_id"]] + 1
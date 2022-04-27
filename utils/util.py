import csv

def read_triples(triples_path):
    queries, pos_passages, neg_passages = [], [], []

    with open(triples_path, 'r', encoding='latin-1') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for query, pos_passage, neg_passage in reader:
            queries.append(query)
            pos_passages.append(pos_passage)
            neg_passages.append(neg_passage)
    
    return {'query': queries, 'pos_passage': pos_passages, 'neg_passage': neg_passages}

def read_qidpidtriples(qidpidtriples_path):
    qids, pos_pids, neg_pids = [], [], []
    with open(qidpidtriples_path, 'r', encoding='latin-1') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for qid, pos_pid, neg_pid in reader:
            qids.append(int(qid))
            pos_pids.append(int(pos_pid))
            neg_pids.append(int(neg_pid))
            
    return {'qid': qids, 'pos_pid': pos_pids, 'neg_pid': neg_pids}

def read_queries(queries_path):
    queries = dict()
    with open(queries_path, 'r', encoding='latin-1') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for qid, query in reader:
            queries[int(qid)] = query
    
    return queries

def read_collection(collection_path):
    collection = dict()
    with open(collection_path, 'r', encoding='latin-1') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for pid, passage in reader:
            collection[int(pid)] = passage

    return collection

def read_qrels(qrels_path):
    qids, pids = [], []
    with open(qrels_path, 'r', encoding='latin-1') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for qid, _, pid, _ in reader:
            qids.append(int(qid))
            pids.append(int(pid))

    return {'qid': qids, 'pid': pids}

def read_top1000(top1000_path):
    qids, pids, queries, passages = [], [], [], []
    with open(top1000_path, 'r', encoding='latin-1') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for qid, pid, query, passage in reader:
            qids.append(qid)
            pids.append(pid)
            queries.append(query)
            passages.append(passage)

    return {'qid': qids, 'pid': pids, 'query': queries, 'passage': passages}

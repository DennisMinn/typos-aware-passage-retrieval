import csv

def read_triples(triples_path):
    triples = []
    with open(triples_path, 'r', encoding='latin-1') as infile:
        for query, pos_passage, neg_passage in reader:
            triples.append({'query': query,
                            'pos_passage': pos_passage,
                            'neg_passage': neg_passage})
    return triples

def read_qidpidtriples(qidpidtriples_path):
    triples = []
    with open(qidpidtriples_path, 'r', encoding='latin-1') as infile:
        for qid, pos_pid, neg_pid in reader:
            triples.append({'qid':qid,
                            'pos_pid':pos_pid,
                            'neg_pid':neg_pid})
    return triples

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
    qrels = []
    with open(qrels_path, 'r', encoding='latin-1') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for qid, _, pid, _ in reader:
            qrels.append({'qid': qid, 'pid': pid})

    return qrels

def read_top1000(top1000_path):
    top1000 = []
    with open(top1000_path, 'r', encoding='latin-1') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for qid, pid, query, passage in reader:
            top1000.append({'qid': qid,
                            'pid': pid,
                            'passage': passage,
                            'query': query})

    return top1000

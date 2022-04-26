import csv
import tqdm as tqdm

from collections import defaultdict

def read_input(chunks = None, chunk_size = 5000, **params):
    #TODO 
    # chunks * chunk_size < total_rows
    # fix tqdm
    # add logs

    df = pd.DataFrame()
    file_path = params['filepath_or_buffer']
    
    print('Counting lines')
    n_rows = chunks * chunk_size if chunks is not None else sum(1 for row in open(file_path, 'rb'))
    params['chunksize'] = chunk_size
    params['low_memory'] = False
    
    print('Parsing lines')
    with tqdm.tqdm(total = n_rows, desc = 'Reading Chunks: ') as progress_bar:
        for i, chunk in enumerate(pd.read_csv(**params)):
            df = pd.concat([df, chunk])
            progress_bar.update(chunk_size)
            
            if(i == chunks):
                break
    return df

def read_triples(path, chunks=None, chunk_size=5000):
    params = {
	'filepath_or_buffer': path, 
	'sep':'\t',
	'header': None,
	'names': ['query', 'positive_passage', 'negative_passage'],
	'compression': 'infer',
    }
    
    return read_input(**params)

def read_qidpidtriples(path):
    triples = []
    with open(qidpidtriples_path, 'r', encoding='latin-1') as infile:
        for qid, pos_pid, neg_pid in reader:
            triples.append({'qid':qid,
                            'pos_pid':pos_id,
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

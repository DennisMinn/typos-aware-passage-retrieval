import pandas as pd
import tqdm as tqdm

def read_input(chunks = None, chunk_size = 5000, **params):
    #TODO 
    # chunks * chunk_size < total_rows
    # fix tqdm
    # add logs

    df = pd.DataFrame()
    file_path = params['filepath_or_buffer']
    
    n_rows = chunks * chunk_size if chunks is not None else sum(1 for row in open(file_path, 'rb'))
    params['chunksize'] = chunk_size
    params['low_memory'] = False
    
    with tqdm.tqdm(total = n_rows, desc = 'Reading Chunks: ') as progress_bar:
        for i, chunk in enumerate(pd.read_csv(**params)):
            df = pd.concat([df, chunk])
            progress_bar.update(chunk_size)
            
            if(i == chunks):
                break
    return df

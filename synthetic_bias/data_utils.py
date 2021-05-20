import os
from pathlib import Path
import requests
import pandas as pd
import zipfile

datasets_config = {'SNLI': {'urlpath': 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
                            'files': ['snli_1.0_train.txt', 'snli_1.0_dev.txt', 'snli_1.0_test.txt'],
                            'fields': ['sentence1', 'sentence2', 'gold_label'],
                            'filters': {'sentence1': [''], 'sentence2': [''], 'gold_label': ['', '-']}},
                   'MNLI': {'urlpath': 'https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip',
                            'files': ['multinli_1.0_train.txt', 'multinli_1.0_dev_matched.txt',
                                      'multinli_1.0_dev_mismatched.txt'],
                            'fields': ['sentence1', 'sentence2', 'gold_label'],
                            'filters': {'sentence1': [''], 'sentence2': [''], 'gold_label': ['', '-']}}
                   }

datafiles_config = {'SNLI_hard': {'urlpath': 'https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl',
                                  'files': ['snli_1.0_test_hard.jsonl'],
                                  'fields': ['sentence1', 'sentence1_binary_parse', 'sentence2',
                                             'sentence2_binary_parse',
                                             'gold_label'],
                                  'filters': {'sentence1': [''], 'sentence2': [''], 'gold_label': ['', '-']}
                                  }
                    }


def prepare_dataset(dataset='SNLI', force=False):
    """
    Prepare dataset for model. Downloads dataset from url, extracts and processes required data files.
    Then writes files to 'data/<dataset>/'.
    Currently only supports zip format datasets. To add datasets add an appropriate entry to the
    dataset config. If force is true and the files already exist, will override the existing files and
    redownload and prepare dataset.
    Returns a list of the file paths.
    """
    cfg = datasets_config[dataset]
    urlpath, files, fields, filters = cfg['urlpath'], cfg['files'], cfg['fields'], cfg['filters']

    sep = os.path.sep
    dataset_dir = sep.join(['data', dataset])
    dataset_path = sep.join([dataset_dir, 'dataset'])
    os.makedirs(dataset_dir, exist_ok=True)

    if all([os.path.isfile(sep.join([dataset_dir, file])) for file in files]) and not force:
        pass
    else:
        # download dataset to data/<dataset_name>/dataset"
        response = requests.get(urlpath, stream=True)
        with open(dataset_path, "wb") as text_file:
            for chunk in response.iter_content(chunk_size=1024):
                text_file.write(chunk)
            text_file.truncate()

        assert zipfile.is_zipfile(dataset_path), "Currently only supports zip format datasets"

        if zipfile.is_zipfile(dataset_path):
            with zipfile.ZipFile(dataset_path) as dataset_zip:
                for file, filename in zip(list(map(lambda x: x.filename, dataset_zip.filelist)),
                                          list(map(lambda x: os.path.split(x.filename)[-1], dataset_zip.filelist))):
                    # unzip required file and write it to 'data/<dataset_name>/<file_name>
                    if filename in files:
                        with dataset_zip.open(file) as zipf:
                            content = zipf.read()
                            with open(sep.join([dataset_dir, filename]), 'wb') as f:
                                f.write(content)
                                f.truncate()

                        # process and overwrite
                        process_tsv(sep.join([dataset_dir, filename]), fields, filters)

    return [sep.join([dataset_dir, file]) for file in files]


def prepare_data_file(datafile='SNLI_hard', force=False):
    """
    Downloads a file from the url, processes it and writes in tsv format to 'data/data_files/<name>.txt.
    Returns the path to the written file. Supports jsonl and tsv file formats.
    """
    cfg = datafiles_config[datafile]
    urlpath, files, fields, filters = cfg['urlpath'], cfg['files'], cfg['fields'], cfg['filters']
    file_type = Path(urlpath).suffix
    filepath = '/'.join(['data', 'datafiles', datafile + file_type])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.isfile('/'.join(['data', 'datafiles', datafile + '.txt'])) and not force:
        pass
    else:
        response = requests.get(urlpath, stream=True)
        with open(filepath, "wb") as text_file:
            for chunk in response.iter_content(chunk_size=1024):
                text_file.write(chunk)
            text_file.truncate()

        process_tsv(filepath, fields, filters)

    return filepath


def process_tsv(filepath, fields, filters):
    """
    Process tab separated file. Overwrites the input file with the processed version.
    :param filepath: path to file to process. Currently only tsv files are supported.
    :param fields: a list of strings indicating the names of fields (columns) to keep
    :param filters: a dictionary. keys are field names, values are a list of values to discard - a sample that matches
    the filter on the specified field will be removed from file.
    """
    # preprocess and save - get only required columns (according to fields)
    # and filter out invalid samples (according to filters)
    # only handles tsv files
    filename, file_extension = os.path.splitext(filepath)
    assert file_extension == '.txt' or file_extension == '.jsonl', "Only .txt and .jsonl files supported"
    with open(filepath, 'r') as f:
        if file_extension == '.txt':
            df = pd.read_table(f, sep='\t+', keep_default_na=False)
        else:
            df = pd.read_json(f, orient='records', lines=True)
    if fields is not None:
        df = df[fields]
    for k in filters.keys():
        df = df[~df[k].isin(filters[k])]
    new_filepath = filename + '.txt'
    os.remove(filepath)
    with open(new_filepath, 'w') as f:
        f.seek(0)
        df.to_csv(f, sep='\t', index=False, line_terminator='\n')
        f.truncate()


if __name__ == "__main__":
    prepare_dataset(dataset='SNLI')
    prepare_data_file('SNLI_hard')

import os
from pathlib import Path
import requests
import pandas as pd
import zipfile
from dataset_utils import lexical_overlap, is_constituent, is_subsequence

datasets_config = {'SNLI': {'urlpath': 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
                            'files': ['snli_1.0_train.txt', 'snli_1.0_dev.txt', 'snli_1.0_test.txt'],
                            'NUM_LABELS': 3,
                            'label_str_to_int': lambda x: {'contradiction': 0, 'entailment': 1, 'neutral': 2}[x],
                            'label_int_to_str': lambda x: ['contradiction', 'entailment', 'neutral'][x],
                            'fields': ['sentence1', 'sentence1_binary_parse', 'sentence2', 'sentence2_binary_parse',
                                       'gold_label', 'label1', 'label2', 'label3', 'label4', 'label5'],
                            'filters': {'sentence1': [''], 'sentence2': [''], 'gold_label': ['', '-']}},
                   'SNLI_Binary': {'urlpath': 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
                                   'files': ['snli_1.0_train.txt', 'snli_1.0_dev.txt', 'snli_1.0_test.txt'],
                                   'NUM_LABELS': 2,
                                   'label_str_to_int': lambda x: {'non-entailment': 0, 'entailment': 1}[x],
                                   'label_int_to_str': lambda x: ['non-entailment', 'entailment'][x],
                                   'fields': ['sentence1', 'sentence1_binary_parse', 'sentence2',
                                              'sentence2_binary_parse',
                                              'gold_label', 'label1', 'label2', 'label3', 'label4', 'label5'],
                                   'filters': {'sentence1': [''], 'sentence2': [''], 'gold_label': ['', '-']},
                                   'mappings': {'contradiction': 'non-entailment',
                                                'entailment': 'entailment', 'neutral': 'non-entailment'}

                                   },
                   'MNLI': {'urlpath': 'https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip',
                            'files': ['multinli_1.0_train.txt', 'multinli_1.0_dev_matched.txt',
                                      'multinli_1.0_dev_mismatched.txt'],
                            'NUM_LABELS': 3,
                            'label_str_to_int': lambda x: {'contradiction': 0, 'entailment': 1, 'neutral': 2}[x],
                            'label_int_to_str': lambda x: ['contradiction', 'entailment', 'neutral'][x],
                            'fields': ['genre', 'sentence1', 'sentence1_binary_parse', 'sentence2',
                                       'sentence2_binary_parse',
                                       'gold_label', 'label1', 'label2', 'label3', 'label4', 'label5'],
                            'filters': {'sentence1': [''], 'sentence2': [''], 'gold_label': ['', '-']}},
                   'MNLI_Binary': {'urlpath': 'https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip',
                                   'files': ['multinli_1.0_train.txt', 'multinli_1.0_dev_matched.txt',
                                             'multinli_1.0_dev_mismatched.txt'],
                                   'NUM_LABELS': 2,
                                   'label_str_to_int': lambda x: {'non-entailment': 0, 'entailment': 1}[x],
                                   'label_int_to_str': lambda x: ['non-entailment', 'entailment'][x],
                                   'fields': ['genre', 'sentence1', 'sentence1_binary_parse', 'sentence2',
                                              'sentence2_binary_parse',
                                              'gold_label', 'label1', 'label2', 'label3', 'label4', 'label5'],
                                   'filters': {'sentence1': [''], 'sentence2': [''], 'gold_label': ['', '-']},
                                   'mappings': {'contradiction': 'non-entailment',
                                                'entailment': 'entailment', 'neutral': 'non-entailment'}

                                   }
                   }

datafiles_config = {
    'HANS': {'urlpath': 'https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt',
             'files': ['heuristics_evaluation_set.txt'],
             'NUM_LABELS': 2,
             'label_str_to_int': lambda x: {'non-entailment': 0, 'entailment': 1}[x],
             'label_int_to_str': lambda x: ['non-entailment', 'entailment'][x],
             'fields': ['pairID', 'gold_label', 'sentence1', 'sentence1_binary_parse', 'sentence2',
                        'sentence2_binary_parse', 'heuristic', 'subcase'],
             'filters': {}
             },
    'SNLI_hard': {'urlpath': 'https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl',
                  'files': ['snli_1.0_test_hard.jsonl'],
                  'NUM_LABELS': 3,
                  'label_str_to_int': lambda x: {'contradiction': 0, 'entailment': 1, 'neutral': 2}[x],
                  'label_int_to_str': lambda x: ['contradiction', 'entailment', 'neutral'][x],
                  'fields': ['sentence1', 'sentence1_binary_parse', 'sentence2', 'sentence2_binary_parse',
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
    mappings = cfg.get('mappings', None)

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
                        process_tsv(sep.join([dataset_dir, filename]), fields, filters, mappings=mappings)

    return [sep.join([dataset_dir, file]) for file in files]


def prepare_data_file(datafile='HANS', force=False):
    """
    Downloads a file from the url, processes it and writes in tsv format to 'data/data_files/<name>.txt.
    Returns the path to the written file. Supports jsonl and tsv file formats.
    """
    cfg = datafiles_config[datafile]
    urlpath, files, fields, filters = cfg['urlpath'], cfg['files'], cfg['fields'], cfg['filters']
    mappings = cfg.get('mappings', None)
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

        process_tsv(filepath, fields, filters, mappings=mappings)

    return filepath


def process_tsv(filepath, fields, filters, mappings=None):
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
    if mappings is not None:
        df.replace(mappings, inplace=True)
    new_filepath = filename + '.txt'
    os.remove(filepath)
    with open(new_filepath, 'w') as f:
        f.seek(0)
        df.to_csv(f, sep='\t', index=False, line_terminator='\n')
        f.truncate()


def generate_heuristics_file(dataset='MNLI_Binary', force=False):
    """
    For each of the train/val/test of a dataset generate a file that holds all the samples containing
    a specified heuristic. Currently only implemented for MNLI_Binary with HANS heuristics.
    The fields of the generated files are:
    'sample_idx', 'sentence1', 'sentence2', 'gold_label', 'lexical_overlap', 'subsequence', 'constituent'.
    :param dataset: Currently only MNLI_Binary, can be extended to SNLI
    :param force: force regeneration of files even if they exist
    :return: paths to generated files
    """
    file_train, file_val, file_test = prepare_dataset(dataset)
    fields = datasets_config[dataset]['fields']
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence1_binary_parse',
                                                                 'sentence2', 'sentence2_binary_parse', 'gold_label']]
    heuristics_filepath_list = []
    os.makedirs('/'.join(['data', 'datafiles']), exist_ok=True)
    for file in datasets_config[dataset]['files']:
        filename, file_extension = os.path.splitext(file)
        heuristics_filepath = '/'.join(['data', 'datafiles', filename + '_heristics' + file_extension])
        heuristics_filepath_list.append(heuristics_filepath)

    for filepath, heuristics_filepath in zip([file_train, file_val, file_test], heuristics_filepath_list):
        if os.path.isfile(heuristics_filepath) and not force:
            continue
        else:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # remove line ending and split on tabs. Skip first line (headers)
            samples = []
            for line in lines[1:]:
                samp = line.splitlines()[0].split(sep='\t')
                candidate = tuple([samp[field_idx] for field_idx in field_indices])
                if len(candidate[0].split()) < 80:
                    samples.append(candidate)

            samples_with_heuristic = []
            for ind, samp in enumerate(samples):
                prem, prem_parse, hyp, hyp_parse, label = samp
                all_in, _ = lexical_overlap(prem, hyp, norm_by_hyp=False)
                subsequence_flag = is_subsequence(prem, hyp)
                constituent_flag = is_constituent(hyp, prem_parse)
                if any([all_in, subsequence_flag, constituent_flag]):
                    flags = (all_in, subsequence_flag, constituent_flag)
                    samples_with_heuristic.append((ind, prem, hyp, label) + flags)

            df = pd.DataFrame(samples_with_heuristic, columns=['sample_idx', 'sentence1', 'sentence2', 'gold_label',
                                                               'lexical_overlap', 'subsequence', 'constituent'])
            with open(heuristics_filepath, 'w') as f:
                f.seek(0)
                df.to_csv(f, sep='\t', index=False, line_terminator='\n')
                f.truncate()

    return heuristics_filepath_list


if __name__ == "__main__":
    prepare_dataset(dataset='SNLI')
    # prepare_dataset(dataset='SNLI_Binary')
    # prepare_dataset(dataset='MNLI')
    prepare_dataset(dataset='MNLI_Binary')
    prepare_data_file('HANS')
    prepare_data_file('SNLI_hard')
    generate_heuristics_file(dataset='MNLI_Binary')
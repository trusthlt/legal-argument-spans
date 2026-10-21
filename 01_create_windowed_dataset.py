"""Create the context-window (0/2/4 previous paragraphs) and full-document
dataset variants from the reformatted data and save them as Hugging Face
DatasetDicts (train/test split by the fixed LAM:ECHR test file ids)."""
import json
from datasets import Dataset, DatasetDict, load_from_disk
import pandas as pd
from utils import find_positions
from collections import Counter
from argparse import ArgumentParser
import os


def create_windowed_dataset(json_data, n:int, min_paragraphs:int=4) -> dict[str, dict[int, dict[str, str|int|list]]]:
    """Generate a dataset from the original input with a context of n previous paragraphs

    Args:
        json_data: Data loaded from json file 
        n (int): Window size, ie how many paragraphs before the current paragraph will be supplied
        min_paragraphs (int, optional): The minimum amount of paragraphs in a file. Used to filter out really short files. Defaults to 4.

    Returns:
        dict[str, dict[int, dict[str, str|int|list]]]: Dictionary with the filename as a key. 
        For each key there is a dictionary with paragraphs ids as a key which contains the content.
        The content contains the following information.
            - 'file_index' (int): numerical index of the current file,
            - 'index' (int): index of the current paragraph,
            - 'file_name' (str): identifier of the current file,
            - 'context' (str): context,
            - 'current_paragraph' (str): the text of the current target paragraph,
            - 'annotated_paragraph' (str): the text of the current target paragraph with in-text annotations,
            - 'positions' (list[tuple[int]|str]): list of all annotations with their start and end position in the reference text, followed by the label
    """

    windowed_data = {}
    current_file = None
    current_paragraphs = []
    file_data = {}

    for key, value in json_data.items():
        file_index, paragraph_index = map(int, key.split('_'))
        file_name, non_annotated_paragraph, annotated_paragraph = value
        if current_file is None:
            current_file = file_index

        if file_index != current_file:
            # Only process files with at least min_paragraphs paragraphs
            if len(current_paragraphs) >= min_paragraphs:
                file_data[current_file] = current_paragraphs
            current_file = file_index
            current_paragraphs = []

        current_paragraphs.append((file_name, non_annotated_paragraph, annotated_paragraph))

    if len(current_paragraphs) >= min_paragraphs:
        file_data[current_file] = current_paragraphs

    for file_index, paragraphs in file_data.items():
        for i in range(len(paragraphs)):
            if i < n:
                # For the first n paragraphs, create entries with no or partial context
                window = paragraphs[max(0, i-n):i+1]
                file_names = [p[0] for p in window]
                original_texts = [p[1] for p in window]
                annotated_texts = [p[2] for p in window]
                context = ' '.join(original_texts[:-1]) if i > 0 else ''  # Partial context or no context
            else:
                # For subsequent paragraphs, create entries with full context
                window = paragraphs[i-n:i+1]
                file_names = [p[0] for p in window]
                original_texts = [p[1] for p in window]
                annotated_texts = [p[2] for p in window]
                context = ' '.join(original_texts[:-1])  # Full context

            positions = find_positions(annotated_texts[-1], original_texts[-1])
            new_entry = {
                'file_index': file_index,
                'index': i,
                'file_name': file_names[-1],
                'context': context,
                'current_paragraph': original_texts[-1],
                'annotated_paragraph': annotated_texts[-1],
                'positions': positions
            }
            if file_names[-1] in windowed_data.keys():
                windowed_data[file_names[-1]][i] = new_entry
            else:
                windowed_data[file_names[-1]] = {}
                windowed_data[file_names[-1]][i] = new_entry
    return windowed_data


def create_full_dataset(json_data) -> dict[int, dict[int|str|list]]:
    """Generate a dataset from the original input staying true to the original data

    Args:
        json_data: Data loaded from json file 

    Returns:
        dict[int, dict[int|str|list]]: Returns a dictionary with with document ids as keys which contains the content.
        The content contains the following information.
            - 'file_index' (int): numerical index of the current file,
            - 'original' (str): the text of the current target file,
            - 'annotated' (str): the text of the current target file with in-text annotations,
            - 'positions' (list[tuple[int]]): list of all annotations with their start and end position in the reference text
            - 'labels' (list[str]): list of all annotated labels, in the order to match the list of positions
    """

    df_docs = pd.DataFrame.from_dict(json_data, orient='index', columns=['file', 'original', 'annotated'])
    full_data = {}
    for i, g in df_docs.groupby('file'):
        orig = ' '.join(g['original'])
        ann = ' '.join(g['annotated'])
        pos_and_labels = find_positions(ann, orig)
        full_data[i] = {
            'file_index': i,
            'original': orig,
            'annotated': ann,
            'positions': [p[0] for p in pos_and_labels],
            'labels': [p[1] for p in pos_and_labels]
            }
    return full_data


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='data/preprocessed_dataset.json')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--window', default=4, type=int)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    
    # `--full` builds the full-document dataset; otherwise a windowed dataset
    # with `--window` previous paragraphs of context is created.
    full = args.full
    preprocessed_file = args.input
    out_dir = args.output_dir
    window_size = args.window

    # sorted test files, ids as defined by LAM:ECHR
    test_files = [
    '001-61184', '001-138992', '001-58274', 
    '001-61747', '001-116186', '001-90975', 
    '001-87567', '001-60492', '001-101152', '001-86253', 
    '001-122716', '001-61002', '001-120959', '001-57504', 
    '001-175140', '001-144661', '001-72205', '001-115868', 
    '001-185495', '001-86233', '001-206503', '001-113460', 
    '001-169661', '001-84106', '001-60791', '001-94206', 
    '001-200324', '001-96585', '001-86008', '001-170362', 
    '001-179427', '001-151051', '001-173490', '001-57519', 
    '001-60696', '001-177082', '001-187926'
    ]

    # Load JSON
    with open(preprocessed_file, 'r') as json_file:
        data = json.load(json_file)

    print(f"loaded {len(data.keys())} documents")

    if full:
        full_data = create_full_dataset(data)

        # Split the dataset into training and test sets based on file names
        train_data = []
        test_data = []

        for k, item in full_data.items():
            if item['file_index'] in test_files:
                continue
            else:
                train_data.append(item)
        
        for tf in test_files:
            test_data.append(full_data[tf])

        # Print the number of examples in train and test sets
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of test examples: {len(test_data)}")

        ds_train = Dataset.from_pandas(pd.DataFrame(data=train_data))
        ds_eval = Dataset.from_pandas(pd.DataFrame(data=test_data))
        ds_lam = DatasetDict({"train": ds_train, "test": ds_eval})

        # Save the DatasetDict as a Hugging Face dataset
        out_path = os.path.join(out_dir, f'full-dataset')
        ds_lam.save_to_disk(out_path)
        print(f"Full dataset saved to {out_path}'")

    else:

        # Create the windowed dataset
        windowed_data = create_windowed_dataset(data, window_size)

        # Split the dataset into training and test sets based on file names
        train_data = []
        test_data = []

        for k, item in windowed_data.items():
            if k in test_files:
                continue
            else:
                for g, v in item.items():
                    train_data.append(v)
        
        for tf in test_files:
            for k, v in windowed_data[tf].items():
                test_data.append(v)

        # Print the number of examples in train and test sets
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of test examples: {len(test_data)}")

        data_train = []
        data_test = []

        for item in train_data:
            pos_and_labels = find_positions(item['annotated_paragraph'], item['current_paragraph'])
            labels = [p[1] for p in pos_and_labels]
            positions = [p[0] for p in pos_and_labels]
            data_train.append({
                'file_index': item['file_index'],
                'index': item['index'],
                'file_name': item['file_name'],
                'original': item['current_paragraph'],
                'annotated': item['annotated_paragraph'],
                'context': item['context'],
                'positions': positions,
                'labels': labels,
            })

        for item in test_data:
            pos_and_labels = find_positions(item['annotated_paragraph'], item['current_paragraph'])
            labels = [p[1] for p in pos_and_labels]
            positions = [p[0] for p in pos_and_labels]
            data_test.append({
                'file_index': item['file_index'],
                'index': item['index'],
                'file_name': item['file_name'],
                'original': item['current_paragraph'],
                'annotated': item['annotated_paragraph'],
                'context': item['context'],
                'positions': positions,
                'labels': labels,
            })

        ds_train = Dataset.from_pandas(pd.DataFrame(data=data_train))
        ds_eval = Dataset.from_pandas(pd.DataFrame(data=data_test))
        ds_lam = DatasetDict({"train": ds_train, "test": ds_eval})

        print(ds_lam)
        out_path = os.path.join(out_dir, f'windowed_dataset-replicate-LAM-{window_size}')
        ds_lam.save_to_disk(out_path)
        print(f"Windowed dataset saved to {out_path}'")

    # Load dataset from disk and display stats
    loaded_dataset = load_from_disk(out_path)

    print("\nTest Dataset Examples:")
    for k, v in loaded_dataset['test'][5].items():
        if type(v) == str:
            print(f"{k}: {v[:100]}")
        else:
            print(f"{k}: {v}")

    counter_train = Counter()
    for s in loaded_dataset['train']:
        annotated_target = s['annotated']
        pos_and_labels = find_positions(annotated_target, s['original'])
        gold_labels = [pair[1] for pair in pos_and_labels]
        counter_train.update(gold_labels)

    counter_test = Counter()
    for s in loaded_dataset['test']:
        annotated_target = s['annotated']
        pos_and_labels = find_positions(annotated_target, s['original'])
        gold_labels = [pair[1] for pair in pos_and_labels]
        counter_test.update(gold_labels)

    train_size = len(loaded_dataset['train'])
    test_size = len(loaded_dataset['test'])
    print(f"Test set distribution of labels:\nn old\t-\t% old\t-\tn new\t-\t% new\t-\tlabel")
    for k, v in counter_train.most_common():
        print(f"{v:5}\t-\t{(v/train_size)*100:4.2f}\t-\t{counter_test[k]:5}\t-\t{(counter_test[k]/test_size)*100:4.2f}\t-\t{k}")

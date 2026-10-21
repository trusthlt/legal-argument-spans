"""Extract paragraphs and legal-argument annotations from the LAM:ECHR XMI gold
data and reformat the BIO annotations into in-text span annotations, saved to
data/preprocessed_dataset.json."""
from cassis import load_typesystem, load_cas_from_xmi
import re
import os
from tqdm import tqdm
import pandas as pd
import json
import configparser
from collections import Counter

from utils import LABEL_MAPPING


def read_xmi(file, typesystem):
    """
    Reads the data from an xmi file and returns the tokens, paragraphs and annotations.
    :param file: path to xmi file
    :param typesystem: typesystem of the xmi
    :return: tokens, paragraphs and annotations
    """
    with open(file, 'rb') as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem)

    tokens = cas.select('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token')
    paragraphs = cas.select('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence')
    annotations = cas.select('webanno.custom.LegalArgumentation')

    return tokens, paragraphs, annotations


def select_text_in_the_law(paragraphs):
    reached_the_law = False
    relevant_pars = []
    for par in paragraphs:
        text = par.get_covered_text()
        if reached_the_law:
            relevant_pars.append(par)
        if ("THE LAW" in text) or ("AS TO THE LAW" in text):
            reached_the_law = True
        
    # the new text should be shorter than the old text
    assert (len(paragraphs) > len(relevant_pars)), f"nothing was shortened"
    # the new text should not be empty
    assert (len(relevant_pars) > 0), f"THE LAW section was not found \n {[p.get_covered_text() for p in paragraphs][-50:]}"
    return relevant_pars

def select_text_before_opinions(paragraphs):
    reached_opinions = False
    relevant_pars = []
    for par in paragraphs:
        text = par.get_covered_text()
        if ('DISSENTING OPINION' in text) or ('OPINION' in text):
            reached_opinions = True
        if not reached_opinions:
            relevant_pars.append(par)
    return relevant_pars


test = [
    '001-61184.xmi', '001-138992.xmi', '001-58274.xmi', 
    '001-61747.xmi', '001-116186.xmi', '001-90975.xmi', 
    '001-87567.xmi', '001-60492.xmi', '001-101152.xmi', '001-86253.xmi', 
    '001-122716.xmi', '001-61002.xmi', '001-120959.xmi', '001-57504.xmi', 
    '001-175140.xmi', '001-144661.xmi', '001-72205.xmi', '001-115868.xmi', 
    '001-185495.xmi', '001-86233.xmi', '001-206503.xmi', '001-113460.xmi', 
    '001-169661.xmi', '001-84106.xmi', '001-60791.xmi', '001-94206.xmi', 
    '001-200324.xmi', '001-96585.xmi', '001-86008.xmi', '001-170362.xmi', 
    '001-179427.xmi', '001-151051.xmi', '001-173490.xmi', '001-57519.xmi', 
    '001-60696.xmi', '001-177082.xmi', '001-187926.xmi'
]

def insert_in_text_annotations(paragraphs, annotations):
    """Insert in-text annotation tokens into each paragraph.

    For every argument overlapping a paragraph, wrap the covered text as
    `<begin> text <end_of_annotation> label <end_of_label>` and return the original
    paragraphs, the annotated paragraphs, and the (start, end, label) positions.
    """
    assert (len(paragraphs) > 0), f"no input text was passed"
    assert (len(annotations) > 0), f"text contains no annotations"

    inital_len = len(paragraphs)
    
    annotated_paragraphs = []
    original_paragraphs = []
    positions = []
    expected_numbers = []
    
    full_original_text = ""
    full_annotated_text = ""
    
    first_number = None

    annotation_covered = {f"{k.begin}-{k.end}": 0 for k in annotations}
    for p in paragraphs:
        begin_par = p.begin
        end_par = p.end
        full_text = p.get_covered_text()
        annotated_text = ''
        curr_begin = begin_par
        rel_annotations = []
        for a in annotations:
            if ((a.begin >= begin_par) and (a.begin <= end_par)) or ((a.end >= begin_par) and (a.end <= end_par)) or ((a.begin <= begin_par) and (a.end >= end_par)):
                # hack for overlapping annotation through multiple paragraphs
                # make sure that annotation begin and end are within the paragraph
                annotation_end = min(a.end, end_par)
                annotation_begin = max(a.begin, begin_par)
                rel_annotations.append(a)
                text_from_par = full_text[annotation_begin-begin_par:annotation_end-begin_par]
                # hack for multiparagraph annotations, only pick the overlap between the paragraph text and the annotation (minimal text)
                covered_text = "".join(filter(lambda i: i in a.get_covered_text(), text_from_par))
                new_text_segment = f"{SPAN_START_TOKEN} {covered_text} {SPAN_SEPARATOR_TOKEN} {LABEL_MAPPING[a.ArgType]} {SPAN_END_TOKEN}"
                positions.append((str(annotation_begin-begin_par), str(annotation_end-begin_par), LABEL_MAPPING[a.ArgType]))
                annotated_text += full_text[curr_begin-begin_par:annotation_begin-begin_par]
                annotated_text += new_text_segment
                curr_begin = min(annotation_end, end_par)
                label_counter.update([LABEL_MAPPING[a.ArgType]])
                annotation_covered[f"{a.begin}-{a.end}"] += 1
        
        # add the remaining text after the last annotation in the paragraph
        annotated_text += full_text[curr_begin-begin_par:end_par]
        
        # Check for numbered paragraph pattern at the beginning of the paragraph
        match = re.match(r'^(\d+)\.', full_text)
        if match:
            number = int(match.group(1))
            if first_number is None:
                first_number = number
                expected_numbers.append(number)
            elif number == expected_numbers[-1] + 1:
                expected_numbers.append(number)
        
        # Concatenate the current paragraph
        full_original_text += " " + full_text
        full_annotated_text += " " + annotated_text

        annotated_paragraphs.append(annotated_text)
        original_paragraphs.append(full_text)

    final_len = len(annotated_paragraphs)

    #assert inital_len == final_len, "not the same len"
    
    return original_paragraphs, annotated_paragraphs, positions


if __name__ == '__main__':

    # Read paths and annotation tokens from config.ini ([FILES] and [PARAMETERS]).
    # The LAM:ECHR gold data is not shipped for licensing reasons; download it and
    # point `gold_data_folder` at the extracted folder.
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    gold_data_folder = config['FILES']['gold_data_folder']

    # Tokens to indicate a new (non-overlapping) annotation in the text
    SPAN_START_TOKEN = config['PARAMETERS']['begin_annotation_token']
    SPAN_SEPARATOR_TOKEN = config['PARAMETERS']['end_annotation_token']
    SPAN_END_TOKEN = config['PARAMETERS']['end_label_token']

    typesystem_path = os.path.join(gold_data_folder, 'TypeSystem.xml')

    with open(typesystem_path, 'rb') as f:
        typesystem = load_typesystem(f)

    # Faulty/unannotated files
    exceptions = ['001-147681.xmi', '001-87934.xmi']

    # Iterate through all files in the directory except for the excluded ones
    test_files = [os.path.join(gold_data_folder, file) for file in os.listdir(gold_data_folder) if file.endswith('.xmi') and (not file in exceptions)]
    print(f"Extracting from {len(test_files)} files.")
    data_test = {}
    label_counter = Counter()
    label_counter_all = Counter()

    file_info = {}
    for file_index, f in enumerate(tqdm(test_files)):
        file_path = f
        counts = []
        docs_tok, docs_para, ans = read_xmi(file_path, typesystem)
        counts.append(len(docs_para))
        selected_pars = select_text_in_the_law(docs_para)
        counts.append(len(selected_pars))
        for a in ans:
            label_counter_all.update([LABEL_MAPPING[a.ArgType]])
        original, annotated, pos = insert_in_text_annotations(selected_pars, ans)
        counts.append(len(original))
        file_info[f.split('/')[-1]] = counts
        for paragraph_index, (orig, annot) in enumerate(zip(original, annotated)):            
            key = f"{file_index}_{paragraph_index}"
            file_name = f.split('/')[-1].split('.')[0]
            data_test[key] = [file_name ,orig, annot]


    df_data = pd.DataFrame.from_dict(data_test, orient='index', columns=['file', 'original', 'annotated'])

    for k, v in file_info.items():
        print(f"{k}: {v[0]} - {v[1]} - {v[2]}")
    for k, v in label_counter.items():
        print(f"{v}\t-{label_counter_all[k]}\t-{k}")

    # Save the results to a JSON file
    output_file = 'data/preprocessed_dataset.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as json_file:
        json.dump(data_test, json_file, indent=4)
        
    print(f"Results saved to {output_file}")
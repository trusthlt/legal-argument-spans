"""Shared helpers: label mapping, in-text span parsing, and the annotation
guidelines used across the data-generation, training and evaluation scripts."""
from nltk.tokenize import RegexpTokenizer
from typing import List
import re

# The annotation tokens are read from config.ini ([PARAMETERS]) so that the whole
# pipeline shares one definition; the bracket tokens are used as a fallback.
import os
import configparser

_config = configparser.ConfigParser()
_config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
if _config.has_section('PARAMETERS'):
    BEGIN_ANNOTATION_TOKEN = _config['PARAMETERS'].get('begin_annotation_token', '[begin_of_annotation]')
    END_ANNOTATION_TOKEN = _config['PARAMETERS'].get('end_annotation_token', '[end_of_annotation]')
    END_LABEL_TOKEN = _config['PARAMETERS'].get('end_label_token', '[end_of_label]')
else:
    BEGIN_ANNOTATION_TOKEN = '[begin_of_annotation]'
    END_ANNOTATION_TOKEN = '[end_of_annotation]'
    END_LABEL_TOKEN = '[end_of_label]'

special_tokens = [BEGIN_ANNOTATION_TOKEN, END_ANNOTATION_TOKEN, END_LABEL_TOKEN]
# match any special token, or a word, or any single non-space character
tokenizer_pattern = '|'.join(re.escape(t) for t in special_tokens) + r'|\w+|\S'
custom_tokenizer = RegexpTokenizer(tokenizer_pattern)

def tokenize_with_special_tokens(text):
    return custom_tokenizer.tokenize(text)

def find_positions_v2(original_text,annotated_paragraph, tokenizer=tokenize_with_special_tokens):
    tokens = tokenizer(annotated_paragraph)
    positions_and_labels = []
    chars_and_labels = []
    i = 0

    while i < len(tokens):
        try:
            begin_pos = tokens.index('[begin_of_annotation]', i)
            try:
                end_pos = tokens.index('[end_of_annotation]', begin_pos)
            except ValueError:
                positions_and_labels.append([(begin_pos, None), "[end_of_annotation] not found"])
                i = begin_pos + 1
                continue

            label_start = end_pos + 1
            try:
                label_end = tokens.index('[end_of_label]', label_start)
                if label_end == label_start:
                    label = "label not found between [end_of_annotation] and [end_of_label]"
                else:
                    label = ' '.join(tokens[label_start:label_end]).strip()
            except ValueError:
                label = "[end_of_label] not found"
                positions_and_labels.append([(begin_pos, end_pos), label])
                i = begin_pos + 1
                continue

            # Check for any token that starts with '[e'
            for token in tokens[label_start:label_end]:
                if token.startswith('[e') and token != '[end_of_label]' and token != '[end_of_annotation]':
                    label = "[end_of_label] generation failed"
                    i = begin_pos + 1
                    continue

            positions_and_labels.append([(begin_pos, end_pos), label])
            first_two, last_two = ' '.join(tokens[begin_pos+1:begin_pos+5]) ,' '.join(tokens[end_pos-4:end_pos])
            # remove extra spaces
            first_two = first_two.strip()
            last_two = last_two.strip()

            first_two = first_two.replace(' .', '.')
            last_two = last_two.replace(' .', '.')

            first_two = first_two.replace(' ,', ',')
            last_two = last_two.replace(' ,', ',')

            last_two = last_two.replace(' )', ')')
            first_two = first_two.replace(' )', ')')

            last_two = last_two.replace(' (', '(')
            first_two = first_two.replace('( ', '(')

            first_two = first_two.replace(' :', ':')
            last_two = last_two.replace(' :', ':')

            first_two = first_two.replace(' ;', ';')
            last_two = last_two.replace(' ;', ';')

            begin_char = original_text.find(first_two)
            end_char = original_text.find(last_two) + len(last_two)
            chars_and_labels.append([(begin_char, end_char), label])
            i = label_end + 1
        except ValueError:
            i += 1
            continue

    return chars_and_labels

def isintokenized(pattern, sequence):
    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i:i+len(pattern)] == pattern:
            return i
    return -1

def find_positions(annotated_paragraph, true_paragraph, tokenizer=tokenize_with_special_tokens):

    # we need to find the positions relative to the input text
    annotation_marker = re.escape(BEGIN_ANNOTATION_TOKEN) + r'(.+?)' + re.escape(END_ANNOTATION_TOKEN) + r'(.+?)' + re.escape(END_LABEL_TOKEN)

    raw_input = re.sub(annotation_marker, r'\1', true_paragraph)
    tokenized_raw = tokenizer(raw_input)

    # replace additional whitespace
    annotated_paragraph = re.sub(r'\s+', ' ', annotated_paragraph)
    raw_input = re.sub(r'\s+', ' ', raw_input)

    positions_and_labels = []
    for valid_annotation in re.finditer(annotation_marker, annotated_paragraph):
        text = valid_annotation.group(1).strip()
        label = valid_annotation.group(2).strip()
        tokenized_text = tokenizer(text)
        if (start_pos := isintokenized(tokenized_text, tokenized_raw)) != -1:
            positions_and_labels.append([(start_pos, start_pos+len(tokenized_text)), label])

    return positions_and_labels


def find_positions_original(annotated_paragraph, tokenizer=tokenize_with_special_tokens):
    tokens = tokenizer(annotated_paragraph)
    #original_tokens = tokenizer(original_paragraph)
    positions_and_labels = []
    i = 0
    marker_token_offset = 0
    label_token_offset = 0

    while i < len(tokens):
        try:
            begin_pos = tokens.index('[begin_of_annotation]', i)
            try:
                end_pos = tokens.index('[end_of_annotation]', begin_pos)
            except ValueError:
                # this means that the label is not valid -> should not be appended to the list
                #positions_and_labels.append([(begin_pos, None), "[end_of_annotation] not found"])
                i = begin_pos + 1
                continue

            label_start = end_pos + 1
            try:
                label_end = tokens.index('[end_of_label]', label_start)
                if label_end == label_start:
                    label = "label not found between [end_of_annotation] and [end_of_label]"
                else:
                    label = ' '.join(tokens[label_start:label_end]).strip()
                    marker_token_offset += 1
            except ValueError:
                label = "[end_of_label] not found"
                #positions_and_labels.append([(begin_pos, end_pos), label])
                i = begin_pos + 1
                continue

            # Check for any token that starts with '[e'
            for token in tokens[label_start:label_end]:
                if token.startswith('[e') and token != '[end_of_label]' and token != '[end_of_annotation]':
                    label = "[end_of_label] generation failed"
                    i = begin_pos + 1
                    continue

            # the first position must be offset (off by 1)
            # for each new annotation span, we substract 2 markers [end of label], [end of annotation] as well as the length of the label
            # this way the token positions correspond to the original input (without actually knowing the original input)
            # this is untested for incomplete annotations
            #label_token_offset += abs(label_end - label_start)
            positions_and_labels.append([(begin_pos - marker_token_offset - label_token_offset + 1, end_pos - marker_token_offset - label_token_offset), label])
            marker_token_offset += 2
            label_token_offset += abs(label_end - label_start)
            i = label_end + 1
        except ValueError:
            i += 1
            continue

    return positions_and_labels

def find_positions_alt(annotated_paragraph, tokenizer=tokenize_with_special_tokens):
    tokens = tokenizer(annotated_paragraph)
    positions_and_labels = []
    awaiting_label = False
    current_label = []
    current_begin = 0
    original_text_index = 0
    for t_i, t in enumerate(tokens):
        if awaiting_label:
            if t == '[end_of_label]':
                awaiting_label = False
                positions_and_labels.append([(current_begin, original_text_index), ' '.join(current_label)])
                current_label = []
            else:
                current_label.append(t)
            continue
        if t == '[end_of_annotation]':
            awaiting_label = True
            continue
        if t == '[begin_of_annotation]':
            current_begin = original_text_index
            continue
        original_text_index += 1

    return positions_and_labels

def get_argument_component_identification(annotated_text, tokenizer=tokenize_with_special_tokens):
    tokens = tokenizer(annotated_text)
    annotated_tokens = tokens
    original_tokens = []
    awaiting_label = False
    for t in tokens:
        if awaiting_label:
            if t == END_LABEL_TOKEN:
                awaiting_label = False
            continue
        if t == END_ANNOTATION_TOKEN:
            awaiting_label = True
            continue
        if t == BEGIN_ANNOTATION_TOKEN:
            continue
        original_tokens.append(t)
    return original_tokens, annotated_tokens

def show_extracted_text(original_text, positions_and_labels, tokenizer=tokenize_with_special_tokens):
    marked_text = []
    tokens = tokenizer(original_text)
    for e in positions_and_labels:
        extract = tokens[e[0][0]:e[0][1]]
        marked_text.append((extract, e[1]))
    return marked_text


def compare_extracted_spans(output_text, target_text, original_text):
    target_annotated = original_text
    output_annotated = original_text

    for line in target_text.split('\n'):
        text = line.rpartition(':')[0].rpartition(']')[0].partition('[')[2]
        label = line.rpartition(':')[2]
        try:
            target_annotated = target_annotated.replace(text, f"[begin_of_annotation] {text} [end_of_annotation] {label} [end_of_label]")
        except:
            print("could not find text")

    for line in output_text.split('\n'):
        text = line.rpartition(':')[0].rpartition(']')[0].partition('[')[2]
        label = line.rpartition(':')[2]
        try:
            output_annotated = output_annotated.replace(text, f"[begin_of_annotation] {text} [end_of_annotation] {label} [end_of_label]")
        except:
            print("could not find text")

    return output_annotated, target_annotated


def text_to_pos_labels(output_text):
    # assume the text is structured like this:
    # [begin;end]: label\n
    l = []
    for line in output_text.split('\n'):
        # [begin;end]: label
        try:
            position = line.rpartition(':')[0].rpartition(']')[0].partition('[')[2]
            begin = position.split(';')[0]
            end = position.split(';')[1]
            label = line.rpartition(':')[2]
            l.append([(begin, end), label])
        except:
            print(f'non conformant line: {line}')
    return l



# maps labels from original German labels to English translation
LABEL_MAPPING = {
        'Subsumtion': 'Application to the concrete case',
        'Entscheidung des EGMR': 'Decision ECHR',
        'Konsens der prozessualen Parteien': 'Non contestation by the parties',
        'Vorherige Rechtsprechung des EGMR': 'Precedents of the ECHR',
        'Verhältnismäßigkeitsprüfung – Rechtsgrundlage': 'Test of the principle of proportionality - Legal basis',
        'Einschätzungsspielraum': 'Margin of appreciation',
        'Verhältnismäßigkeitsprüfung – Angemessenheit': 'Test of the principle of proportionality - Proportionality',
        'Verhältnismäßigkeitsprüfung – Legitimer Zweck': 'Test of the principle of proportionality - Legitimate purpose',
        'Wortlaut Auslegung': 'Textual interpretation',
        'Distinguishing': 'Distinguishing',
        'Sinn & Zweck Auslegung': 'Teleological interpretation',
        'Overruling': 'Overruling',
        'Systematische Auslegung': 'Systematic interpretation',
        'Verhältnismäßigkeitsprüfung – Geeignetheit': 'Test of the principle of proportionality - Suitability',
        'Rechtsvergleichung': 'Comparative law',
    }

ANNOTATION_GUIDELINES = """
###Annotation Guidelines

These guideline annotations serve as an instruction manual for the annotation of ECHR judgments. Judgments of the ECHR shall be annotated according to this guideline. Thereby, it is in the nature of things that a classification on the basis of the categories provided in the guideline can only be made based on a critical appraisal of the full argumentation of the ECHR.
It has to be distinguished between the fifteen possible types of legal arguments.
In many cases, the category of a paragraph of the decision does not result exclusively from the text of that paragraph itself, but only in connection with further paragraphs, so that each paragraph must always be read in context, i.e. in relation to what has been addressed in a previous one.

1. "Non contestation by the parties": Procedural arguments are generally marked as such due to their special nature. This means that even in the part of the judgment on the application to the concrete case, the relevant sentences are not marked as such – although they are nevertheless part of this section – but are grouped into the category of procedural arguments (Non contestation by the parties).
2. "Textual interpretation": The wording is the first indication, but not a rigid boundary for the regulatory content of a norm (which can go beyond its wording). The textual interpretation is complemented by other methods of interpretation (see below). It can be referred to the meaning of the norm wording at the time of its origin or its application considering technical or (most subsidiarily) colloquial language. According to the final clause of the ECHR, only English and French are “authentic” languages, i.e. only these are to be used (other languages only subsidiarily) for the interpretation.
3. "Systematic interpretation": Systematic interpretation is based on the ideal (!) of an in itself consistent legal system. Each legal norm is thus “to be interpreted only from its position and function within the complete legal system”. On European level the relevant law/contract itself, the overall legal order or other international treaties as well as distant influences such as a constitution, the Charter of fundamental human rights, etc. can be taken into account.
4. "Teleological interpretation": Moreover, the category “intent and purpose” includes three further subcategories: the teleological interpretation, the efficiency of the protection (Art. 33 para. 4 VCLT) as well as the (judicial) development of the law. It is controversial if the teleological interpretation is a mean of interpreting a norm or the goal of the interpretation itself. It brings up the question which objective (telos) is to be achieved by the legal norm? The decisive factor is not the historical intention of the legislator, but the objective purpose expressed in the norm. The objective of the norm is characterized significantly by the wording, the systematic and the history (means of interpretation). Regarding the ECHR the teleological interpretation is specified as a „dynamic“ or „evolutive“ interpretation. It takes into account the specialties of the ECHR as a “living instrument, which must be interpreted in the light of present day conditions”, i.e. gives the judges a bigger margin of appreciation.
5. "Comparative law": Legal situation in the Contracting parties/Legal situation in the EU/Autonomous definitions. References to the case law of other courts belong here as well.

Only the following categories (1.-4.) are to be used when there is a proportionality test. At the end, there is a decision of the ECtHR (cf. under C), which is to be annotated accordingly as “decision of the ECtHR”. Occasionally, however, the category “application to the concrete case” may also be used for the legal basis (1.) as well as the legitimate purpose (2.). This depends on the respective individual case and must be assessed critically. For all other points (3.-4.) the categories specified here have to be used while the category “application to the concrete case” is never used.
6. "Test of the principle of proportionality - Legal basis": “In a constitutional democracy, a constitutional right cannot be limited unless such a limitation is authorized by law. This is the principle of legality. From here stems the requirement – which can be found in modern constitutions’ limitation clauses, as well as in other international documents – that any limitation on a right be “prescribed by law”. At the basis of this requirement stands the principle of the rule of law”
7. "Test of the principle of proportionality - Legitimate purpose": “The proper purpose component examines whether a law (a statute or the common law) that limits a constitutional right is for a purpose that justifies such limitation”
8. "Test of the principle of proportionality - Suitability": “The requirement is that the means used by the limiting law fit (or a rational connected to) the purpose the limiting law was designed to fulfill. The requirement is that the means used by the limiting law can realize or advance the underlying purpose of that law; that the use of such means would rationally lead to the realization of the law’s purpose. It is therefore required that the means chosen be pertinent to the realization of the purpose in the sense that the limiting law increases the likelihood of realizing its purpose”. The means used must at least further the achievement of the legitimate purpose.
9. "Test of the principle of proportionality - Proportionality": Since the ECHR – in contrast for instance to the Federal Constitutional Court – does not strictly differentiate between the categories of necessity and proportionality in a strict sense, considerations of necessity – if present – are annotated within this category. “The next component of proportionality is the necessity test. It is also referred to as the requirement of “the less restrictive means”. According to this test, the legislator has to choose – of all those means that may advance the purpose of the limiting law – that which would least limit the human right in question”. The suitable means must be necessary to achieve the legitimate purpose, that is the least restrictive of all equally effective means available. “According to proportionality stricto sensu, in order to justify a limitation on a constitutional right, a proper relation (“proportional” in the narrow sense of the term) should exist between benefits gained by the public and harm caused to the constitutional right from obtaining that purpose. This test requires a balancing of the benefits gained by the public and the harm caused to the constitutional right through the use of the means selected by law to obtain the proper purpose”. In an assessment of the benefits of the measure and the impairment of the affected persons, it must be determined whether the applied measures are appropriate, meaning reasonable for the persons concerned.

10. "Overruling": Overruling is referred to the (re-)adjustment of a precedent on a horizontal level, only under the premise of fundamental deficits of the previous precedent.
11. "Distinguishing": Distinguishing happens if an essential difference of facts is assessed by the judges, which leads to a non-transfer of a precedent to the new case.
12. "Margin of appreciation": The margin of appreciation is a margin of discretion granted by the ECtHR to the judiciary, legislature and executive of the Member States before a violation of the ECtHR is assumed.
13. "Precedents of the ECtHR": Binding effect of the legal content of earlier judgments of the ECtHR for later judgments. Only decisions (of all kind: GC, Chamber, Committee, Commission) of the ECtHR itself belong in this category.
14. "Application to the concrete case": Determination of the relation between the concrete case and the abstract legal norm. Subsumption of the facts of the case under a legal norm, i.e. examination whether the offence is fulfilled and the legal consequence thereby triggered.
15. "Decision ECtHR": The final sentence of the interpretation of a norm as well as the final sentence of the part of the judgment on the application to the concrete case may be concerned. If a section on the application of the Convention to the concrete case presents a pure reproduction of the facts – even though this extends over several paragraph - this is also marked as “application”. This category is generally to be understood broadly.
"""

AG = """These guideline annotations serve as an instruction manual for the annotation of ECtHR judgments. Judgments of the ECtHR shall be annotated according to this guideline. Thereby, it is in the nature of things that a classification on the basis of the categories provided in the guideline can only be made based on a critical appraisal of the full argumentation of the ECtHR.
It has to be distinguished between the fifteen possible types of legal arguments.
In many cases, the category of a paragraph of the decision does not result exclusively from the text of that paragraph itself, but only in connection with further paragraphs, so that each paragraph must always be read in context, i.e. in relation to what has been addressed in a previous one.

1. "Non contestation by the parties": Procedural arguments are generally marked as such due to their special nature. This means that even in the part of the judgment on the application to the concrete case, the relevant sentences are not marked as such – although they are nevertheless part of this section – but are grouped into the category of procedural arguments (Non contestation by the parties).
2. "Textual interpretation": The wording is the first indication, but not a rigid boundary for the regulatory content of a norm (which can go beyond its wording). The textual interpretation is complemented by other methods of interpretation (see below). It can be referred to the meaning of the norm wording at the time of its origin or its application considering technical or (most subsidiarily) colloquial language. According to the final clause of the ECtHR, only English and French are “authentic” languages, i.e. only these are to be used (other languages only subsidiarily) for the interpretation.
3. "Systematic interpretation": Systematic interpretation is based on the ideal (!) of an in itself consistent legal system. Each legal norm is thus “to be interpreted only from its position and function within the complete legal system”. On European level the relevant law/contract itself, the overall legal order or other international treaties as well as distant influences such as a constitution, the Charter of fundamental human rights, etc. can be taken into account.
4. "Teleological interpretation": Moreover, the category “intent and purpose” includes three further subcategories: the teleological interpretation, the efficiency of the protection (Art. 33 para. 4 VCLT) as well as the (judicial) development of the law. It is controversial if the teleological interpretation is a mean of interpreting a norm or the goal of the interpretation itself. It brings up the question which objective (telos) is to be achieved by the legal norm? The decisive factor is not the historical intention of the legislator, but the objective purpose expressed in the norm. The objective of the norm is characterized significantly by the wording, the systematic and the history (means of interpretation). Regarding the ECtHR the teleological interpretation is specified as a „dynamic“ or „evolutive“ interpretation. It takes into account the specialties of the ECtHR as a “living instrument, which must be interpreted in the light of present day conditions”, i.e. gives the judges a bigger margin of appreciation.
5. "Comparative law": Legal situation in the Contracting parties/Legal situation in the EU/Autonomous definitions. References to the case law of other courts belong here as well.

Only the following categories (6.-9.) are to be used when there is a proportionality test. At the end, there is a decision of the ECtHR (cf. under C), which is to be annotated accordingly as “decision of the ECtHR”. Occasionally, however, the category “application to the concrete case” may also be used for the legal basis (1.) as well as the legitimate purpose (2.). This depends on the respective individual case and must be assessed critically. For all other points (3.-4.) the categories specified here have to be used while the category “application to the concrete case” is never used.
6. "Test of the principle of proportionality - Legal basis": “In a constitutional democracy, a constitutional right cannot be limited unless such a limitation is authorized by law. This is the principle of legality. From here stems the requirement – which can be found in modern constitutions’ limitation clauses, as well as in other international documents – that any limitation on a right be “prescribed by law”. At the basis of this requirement stands the principle of the rule of law”
7. "Test of the principle of proportionality - Legitimate purpose": “The proper purpose component examines whether a law (a statute or the common law) that limits a constitutional right is for a purpose that justifies such limitation”
8. "Test of the principle of proportionality - Suitability": “The requirement is that the means used by the limiting law fit (or a rational connected to) the purpose the limiting law was designed to fulfill. The requirement is that the means used by the limiting law can realize or advance the underlying purpose of that law; that the use of such means would rationally lead to the realization of the law’s purpose. It is therefore required that the means chosen be pertinent to the realization of the purpose in the sense that the limiting law increases the likelihood of realizing its purpose”. The means used must at least further the achievement of the legitimate purpose.
9. "Test of the principle of proportionality - Proportionality": Since the ECtHR – in contrast for instance to the Federal Constitutional Court – does not strictly differentiate between the categories of necessity and proportionality in a strict sense, considerations of necessity – if present – are annotated within this category. “The next component of proportionality is the necessity test. It is also referred to as the requirement of “the less restrictive means”. According to this test, the legislator has to choose – of all those means that may advance the purpose of the limiting law – that which would least limit the human right in question”. The suitable means must be necessary to achieve the legitimate purpose, that is the least restrictive of all equally effective means available. “According to proportionality stricto sensu, in order to justify a limitation on a constitutional right, a proper relation (“proportional” in the narrow sense of the term) should exist between benefits gained by the public and harm caused to the constitutional right from obtaining that purpose. This test requires a balancing of the benefits gained by the public and the harm caused to the constitutional right through the use of the means selected by law to obtain the proper purpose”. In an assessment of the benefits of the measure and the impairment of the affected persons, it must be determined whether the applied measures are appropriate, meaning reasonable for the persons concerned.

10. "Overruling": Overruling is referred to the (re-)adjustment of a precedent on a horizontal level, only under the premise of fundamental deficits of the previous precedent.
11. "Distinguishing": Distinguishing happens if an essential difference of facts is assessed by the judges, which leads to a non-transfer of a precedent to the new case.
12. "Margin of appreciation": The margin of appreciation is a margin of discretion granted by the ECtHR to the judiciary, legislature and executive of the Member States before a violation of the ECtHR is assumed.
13. "Precedents of the ECtHR": Binding effect of the legal content of earlier judgments of the ECtHR for later judgments. Only decisions (of all kind: GC, Chamber, Committee, Commission) of the ECtHR itself belong in this category.
14. "Application to the concrete case": Determination of the relation between the concrete case and the abstract legal norm. Subsumption of the facts of the case under a legal norm, i.e. examination whether the offence is fulfilled and the legal consequence thereby triggered.
15. "Decision ECtHR": The final sentence of the interpretation of a norm as well as the final sentence of the part of the judgment on the application to the concrete case may be concerned. If a section on the application of the Convention to the concrete case presents a pure reproduction of the facts – even though this extends over several paragraph - this is also marked as “application”. This category is generally to be understood broadly.
"""

def unbracket(text, token_begin='[', token_end=']'):
    stack = []
    assert len(token_begin) == len(token_end), 'Tokens do not have the same length'
    token_len = len(token_begin)
    skip = 0
    for i, c in enumerate(text):
        if skip > 0:
            skip -= 1
            continue
        if c in token_begin:
            skip = token_len - 1
            stack.append(i)
        elif c in token_end and stack:
            skip = token_len - 1
            start = stack.pop()
            yield (len(stack), text[start + 1 + (token_len-1): i], start+1+(token_len-1), i)


def labeled_output_sentence(output_sentence, with_label=False, token_begin='[', token_end=']', token_sep='|'):
    DEFINE = token_sep
    unbracketed_sentences = unbracket(output_sentence, token_begin=token_begin, token_end=token_end)
    # section the sentence into spans (of arguments), labels (corresponding to the spans)
    # and positions, indicating the relative character positions of the span begin and end
    spans, output_labels, positions = [], [], []
    for level, t, b_idx, e_idx in list(unbracketed_sentences):
        if level == 0:
            positions.append((b_idx, e_idx))
            if with_label:
                sen = t.split(' ')
                if DEFINE in sen:
                    l_idx = t.index(DEFINE)
                    new_text = t[:l_idx]
                    label = t[l_idx+1:e_idx-b_idx].strip()
                    spans.append(new_text)
                    output_labels.append(label)
            else:
                new_text = t
                label = 'I'
                spans.append(new_text)
                output_labels.append(label)

    # return the split sentence and the list of output labels
    return spans, output_labels, positions

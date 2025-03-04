import json
import os
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from typing import Dict, List, Set, Tuple, Any, Optional, Union

_LOG_FILE = None


def ensure_directory(directory_path):
    """ Ensure it exists."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def get_log_file(base_dir="logs"):
    """Grabs the log file or makes a new one if it doesn't exist yet."""
    global _LOG_FILE
    if _LOG_FILE is None:
        ensure_directory(base_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(base_dir, f"story_processing_{timestamp}.jsonl")
        _LOG_FILE = open(log_filename, "a", encoding="utf-8")
    return _LOG_FILE


def log_story_to_jsonl(story_data: Dict[str, Any]):
    """
    Writes story processing info to the log.
    This is for metrics data later and for if the program crashes.
    """
    log_file = get_log_file()

    log_entry = {
        "file_path": story_data["file_path"],
        "culture": story_data["culture"],
        "continent": story_data["continent"],
        "word_count": story_data["word_count"],
        "num_species": len(story_data["species_mentions"]),
        "predator_count": story_data.get("predator_count", 0),
        "prey_count": story_data.get("prey_count", 0)
    }

    log_file.write(json.dumps(log_entry) + "\n")
    log_file.flush()


def close_log_file():
    """^^^what the method says."""
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.close()
        _LOG_FILE = None


def load_data(flora_fauna_path: str):
    """Loads up all the flora and fauna from the CSVs."""
    lemmatizer = WordNetLemmatizer()

    data = {key: {} for key in [
        "trees", "plants", "mammals", "birds",
        "reptiles", "marine", "small_creatures", "mollusc"
    ]}

    lemma_to_original = {}
    original_to_lemma = {}
    multi_word_terms = set()
    species_flags = {}

    file_mapping = {
        "Trees.csv": "trees",
        "Plants.csv": "plants",
        "Mammals.csv": "mammals",
        "Birds.csv": "birds",
        "Reptiles.csv": "reptiles",
        "Marine.csv": "marine",
        "Small_Creatures.csv": "small_creatures",
        "Mollusc.csv": "mollusc"
    }

    for file, category in file_mapping.items():
        file_path = Path(flora_fauna_path) / file
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path).dropna(axis=1, how="all")
        if df.empty:
            continue

        predator_col, prey_col = None, None
        for col_idx, col_name in enumerate(df.columns):
            col_name_clean = col_name.lower().strip()
            if re.search(r"predator", col_name_clean, re.IGNORECASE):
                predator_col = col_idx
            if re.search(r"prey", col_name_clean, re.IGNORECASE):
                prey_col = col_idx

        for _, row in df.iterrows():
            name = str(row.iloc[0]).strip()

            if not name or name.lower() in {"yes", "drill", "nan", "none", "as"}:
                continue

            original_lower = name.lower()
            #this is for donkeys, ok? IDK why but the system always
            #refuses to tag them and they use 'ass' everywhere in the egyptian texts
            if original_lower == "ass":
                continue
            elif original_lower in {"yew", "yews"}:
                lemma_name = "yew"
            elif original_lower in {"mandrill", "mandrills"}:
                lemma_name = "mandrill"
            else:
                if " " in original_lower:
                    multi_word_terms.add(original_lower)
                    lemma_name = original_lower
                else:
                    lemma_name = lemmatizer.lemmatize(original_lower, pos='n')
                    if lemma_name == "as":
                        continue

            data[category][lemma_name] = name
            lemma_to_original[lemma_name] = name
            original_to_lemma[name] = lemma_name

            if lemma_name not in species_flags:
                species_flags[lemma_name] = {}

            if predator_col is not None and len(row) > predator_col:
                predator_value = str(row.iloc[predator_col]).strip().lower()
                species_flags[lemma_name]["predator"] = predator_value in ["yes", "y", "true", "1"]

            if prey_col is not None and len(row) > prey_col:
                prey_value = str(row.iloc[prey_col]).strip().lower()
                species_flags[lemma_name]["prey"] = prey_value in ["yes", "y", "true", "1"]

    return data, lemma_to_original, original_to_lemma, multi_word_terms, species_flags


def should_annotate(species_name: str, word_pos: str) -> bool:
    """This is for quirks in the system with pos tagging and other"""
    species_lower = species_name.lower()

    if species_lower in ["raven", "tree"]:
        return True

    if species_lower == "marten" and word_pos in ("NNP", "NNPS"):
        return False
    if species_lower in ["swift", "sage", "coral"] and word_pos.startswith("JJ"):
        return False

    return word_pos.startswith("NN") or word_pos.startswith("JJ")


def extract_mention_context(sentences, curr_idx, sentence, start, end, matched_text, species_name):
    """Context around the mention for label studio"""
    prev_sentence = sentences[curr_idx - 1][1] if curr_idx > 0 else ""
    next_sentence = sentences[curr_idx + 1][1] if curr_idx < len(sentences) - 1 else ""

    return {
        "species": species_name,
        "context": sentence,
        "full_sentence": sentence,
        "mention_form": matched_text,
        "sentence_position": curr_idx,
        "prev_sentence": prev_sentence,
        "next_sentence": next_sentence,
        "context_window": f"{prev_sentence} {sentence} {next_sentence}".strip(),
        "char_offset": start,
        "char_end": end
    }


def get_species_mentions(
        text: str,
        sentences: List[str],
        indexed_sentences: List[Tuple[int, str]],
        lemma_to_original: Dict[str, str],
        original_to_lemma: Dict[str, str],
        multi_word_terms: Set[str],
        lemmatizer: WordNetLemmatizer,
        output_format: str = "analysis"
) -> Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    A wild program hunting for plants and animals in the great wilderness of text.
    """
    lemmatizer = lemmatizer or WordNetLemmatizer()
    text_lower = text.lower()
    found_positions = set()

    analysis_mentions = defaultdict(list) if output_format == "analysis" else None
    annotation_mentions = [] if output_format == "annotation" else None

    sentence_mapping = {}
    current_pos = 0

    for sent in sentences:
        sent_start = text.find(sent, current_pos)
        if sent_start != -1:
            sent_end = sent_start + len(sent)
            words = word_tokenize(sent)
            pos_tags_ = pos_tag(words)

            sentence_mapping[sent_start] = {
                'text': sent,
                'end': sent_end,
                'pos_tags': pos_tags_
            }
            current_pos = sent_end

    def process_mention(start, end, match_text, species_name):
        sent_start_candidates = [pos for pos in sentence_mapping.keys() if pos <= start]
        if not sent_start_candidates:
            return False

        sent_start = max(sent_start_candidates)
        sent_info = sentence_mapping[sent_start]

        curr_idx = -1
        for i, (_, sentence) in enumerate(indexed_sentences):
            if sentence == sent_info['text']:
                curr_idx = i
                break

        if curr_idx == -1:
            return False

        is_multi_word = " " in match_text.lower()
        special_case = species_name.lower() in ["raven", "tree"]

        word_pos = None
        if not (is_multi_word or special_case):
            first_word = match_text.split()[0]
            for w, pos_ in sent_info['pos_tags']:
                if w.lower() == first_word.lower():
                    word_pos = pos_
                    break

            if not word_pos:
                return False

            if species_name.lower() == "marten" and word_pos in ("NNP", "NNPS"):
                return False
            if species_name.lower() in ["swift", "sage", "coral"] and word_pos.startswith("JJ"):
                return False

            if not (word_pos.startswith("NN") or word_pos.startswith("JJ")):
                return False

        if output_format == "analysis":
            mention_data = extract_mention_context(
                indexed_sentences, curr_idx, sent_info['text'],
                start - sent_start, (start - sent_start) + len(match_text),
                match_text, species_name
            )
            analysis_mentions[species_name].append(mention_data)

        elif output_format == "annotation":
            pos_value = "COMPOUND" if is_multi_word else (
                "nn" if special_case else word_pos
            )

            annotation_mentions.append({
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "text": match_text,
                    "sentence": sent_info['text'],
                    "pos": pos_value,
                    "labels": [species_name]
                }
            })

        return True

    if multi_word_terms:
        multi_word_pattern = '|'.join(re.escape(term) for term in multi_word_terms)
        for match in re.finditer(rf"\b({multi_word_pattern})\b", text_lower):
            start, end = match.span()
            if not any((s <= start < e or s < end <= e) for s, e in found_positions):
                matched_text = match.group(1)
                species_original = lemma_to_original.get(matched_text, matched_text)

                if process_mention(start, end, text[start:end], species_original):
                    found_positions.add((start, end))

    compound_pattern = '|'.join(
        f"{lemma}\\s+(?:tree|fish|flower|rose|marten|bird|slug|snail|etc)s?"
        for lemma in lemma_to_original.keys()
        if " " not in lemma
    )

    if compound_pattern:
        for match in re.finditer(rf"\b({compound_pattern})\b", text_lower):
            start, end = match.span()
            if not any((s <= start < e or s < end <= e) for s, e in found_positions):
                matched_str = match.group(1)
                first_tok = matched_str.split()[0]
                lemma = lemmatizer.lemmatize(first_tok.lower(), pos='n')

                if lemma in lemma_to_original:
                    species_original = lemma_to_original[lemma]

                    if process_mention(start, end, text[start:end], species_original):
                        found_positions.add((start, end))

    offset_in_text = 0
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    for i, (tok, pos_) in enumerate(tagged_tokens):
        tok_lower = tok.lower()

        found_idx = text_lower.find(tok_lower, offset_in_text)

        if found_idx == -1:
            continue

        start = found_idx
        end = found_idx + len(tok)
        offset_in_text = end

        lemma = lemmatizer.lemmatize(tok_lower, pos='n')

        if lemma in lemma_to_original:
            original_name = lemma_to_original[lemma]

            if not any((s <= start < e_ or s < end <= e_) for s, e_ in found_positions):
                if process_mention(start, end, text[start:end], original_name):
                    found_positions.add((start, end))

    return analysis_mentions if output_format == "analysis" else annotation_mentions


def batch_process(items, batch_size, process_func, *args, **kwargs):
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch, *args, **kwargs)
        results.extend(batch_results)

    return results
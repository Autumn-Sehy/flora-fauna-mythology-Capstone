import json
import os
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from typing import Dict, List, Set, Tuple, Any, Union

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

    data = {key: {} for key in [
        "trees", "plants", "mammals", "birds",
        "reptiles", "marine", "small_creatures", "mollusc"
    ]}

    original_to_lower = {}
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
        if file_path.exists():
            df = pd.read_csv(file_path).dropna(axis=1, how="all")

            if not df.empty:
                predator_col, prey_col = None, None
                for col_idx, col_name in enumerate(df.columns):
                    col_name_clean = col_name.lower().strip()
                    if re.search(r"predator", col_name_clean, re.IGNORECASE):
                        predator_col = col_idx
                    if re.search(r"prey", col_name_clean, re.IGNORECASE):
                        prey_col = col_idx

                for _, row in df.iterrows():
                    name = str(row.iloc[0]).strip()

                    #if name and name.lower() not in {"yes", "drill", "nan", "none", "as"}:
                    #this is from an earlier version of the code where I was using lemmatization
                    #I have it on here in case I want to bring that back
                    original_lower = name.lower()

                    if " " in original_lower:
                        multi_word_terms.add(original_lower)

                    data[category][original_lower] = name
                    original_to_lower[name] = original_lower

                    if original_lower not in species_flags:
                        species_flags[original_lower] = {}

                    if predator_col is not None:
                        predator_value = str(row.iloc[predator_col]).strip().lower()
                        species_flags[original_lower]["predator"] = predator_value in ["yes", "y", "true", "1"]

                    if prey_col is not None:
                        prey_value = str(row.iloc[prey_col]).strip().lower()
                        species_flags[original_lower]["prey"] = prey_value in ["yes", "y", "true", "1"]

    return data, original_to_lower, multi_word_terms, species_flags


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
        original_to_lower: Dict[str, str],
        multi_word_terms: Set[str],
        output_format: str = "analysis"
) -> Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    A wild program hunting for plants and animals in the great wilderness of text.
    """
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
        if sent_start_candidates:
            sent_start = max(sent_start_candidates)
            sent_info = sentence_mapping[sent_start]

            curr_idx = next((i for i, (_, sentence) in enumerate(indexed_sentences) if sentence == sent_info['text']), -1)

            if curr_idx != -1:
                is_multi_word = " " in match_text.lower()
                special_case = species_name.lower() in ["raven", "tree"]

                word_pos = None
                if not (is_multi_word or special_case):
                    first_word = match_text.split()[0]
                    word_pos = next((pos_ for w, pos_ in sent_info['pos_tags'] if w.lower() == first_word.lower()), None)

                    if word_pos:
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
        return False

    for term in multi_word_terms:
        for match in re.finditer(rf"\b{re.escape(term)}\b", text_lower):
            start, end = match.span()
            if not any(s <= start < e or s < end <= e for s, e in found_positions):
                species_original = original_to_lower.get(match_text, match_text)
                if process_mention(start, end, text[start:end], species_original):
                    found_positions.add((start, end))

    offset_in_text = 0
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    for tok, pos_ in tagged_tokens:
        tok_lower = tok.lower()
        found_idx = text_lower.find(tok_lower, offset_in_text)

        if found_idx != -1:
            start = found_idx
            end = found_idx + len(tok)
            offset_in_text = end

            original_name = original_to_lower.get(tok_lower)

            if original_name and not any((s <= start < e_ or s < end <= e_) for s, e_ in found_positions):
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
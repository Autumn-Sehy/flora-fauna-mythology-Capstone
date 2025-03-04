import os
from typing import Dict, List, Optional, Any
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from utils import log_story_to_jsonl, get_species_mentions, load_data


class DataProcessor:
    """
    This is for processing all those stories and finding mentions of flora and fauna.
    Then we do sentiment and emotion analysis on them to see how people feel about living creatures.
    """

    def __init__(self, config):
        """
        Sets up the DataProcessor.

        Args:
            config: All the settings and parameters for our processing
        """
        self.config = config
        self.lemmatizer = WordNetLemmatizer()

        self._sentiment_pipeline = None
        self._emotion_pipeline = None

        print("🦊 Loading flora and fauna data...")
        self.flora_fauna_data, self.lemma_to_original, self.original_to_lemma, \
            self.multi_word_terms, self.species_flags = load_data(self.config.FLORA_FAUNA_PATH)
        print("🦝 Data loaded successfully!")

    @property
    def sentiment_pipeline(self):
        """Sometimes we don't want to run the sentiment pipeline because it's expensive."""
        if self._sentiment_pipeline is None and self.config.USE_SENTIMENT_ANALYSIS:
            from transformers import pipeline
            print("🦁 Loading sentiment pipeline...")
            self._sentiment_pipeline = pipeline("sentiment-analysis")
        return self._sentiment_pipeline

    @property
    def emotion_pipeline(self):
        """Sometimes we don't want to run the expensive emotion pipeline."""
        if self._emotion_pipeline is None and self.config.USE_EMOTION_ANALYSIS:
            from transformers import pipeline
            print("🐼 Loading emotion pipeline...")
            self._emotion_pipeline = pipeline("text-classification",
                                              model=self.config.EMOTION_MODEL)
        return self._emotion_pipeline

    def run_sentiment_analysis(self, text: str) -> Optional[Dict[str, Any]]:
        """Figures out if the text is positive or negative."""
        if not self.sentiment_pipeline:
            return None

        text_for_pipeline = text[:self.config.MAX_SEQUENCE_LENGTH]
        result = self.sentiment_pipeline(text_for_pipeline)[0]

        score = result["score"]
        if result["label"] == "NEGATIVE":
            score = -score
        result["score"] = score

        return result

    def run_emotion_analysis(self, text: str) -> Optional[Dict[str, Any]]:
        """Gets the emotions from text - like joy, anger, sadness, the need to overthrow the kingdom."""
        if not self.emotion_pipeline:
            return None

        text_for_pipeline = text[:self.config.MAX_SEQUENCE_LENGTH]
        result = self.emotion_pipeline(text_for_pipeline)[0]

        return result

    def process_story(self, file_path: str, culture: str, continent: str) -> Optional[Dict[str, Any]]:
        """
        Takes a story and finds all the animals and plants mentioned in it.

        Args:
            file_path: Where the story file is located
            culture: What culture the story comes from
            continent: Which continent it's from

        Returns:
            A dictionary with all the analysis/None if something broke (whelp)
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read().strip()

            if not raw_text:
                return None

            words = word_tokenize(raw_text)
            word_count = len(words)

            sentences = sent_tokenize(raw_text)
            indexed_sentences = list(enumerate(sentences))

            # find all the critters in our text
            species_mentions = get_species_mentions(
                raw_text, sentences, indexed_sentences,
                self.lemma_to_original, self.original_to_lemma,
                self.multi_word_terms, self.lemmatizer,
                output_format="analysis"
            )

            for curr_idx, sentence in indexed_sentences:
                sentence_has_creatures = any(
                    mention["sentence_position"] == curr_idx
                    for mentions in species_mentions.values()
                    for mention in mentions
                )

                if sentence_has_creatures:
                    sentiment_result = self.run_sentiment_analysis(sentence)
                    emotion_result = self.run_emotion_analysis(sentence)

                    for sp_name, mention_list in species_mentions.items():
                        for mention in mention_list:
                            if mention["sentence_position"] == curr_idx:
                                mention["sentiment"] = sentiment_result
                                mention["emotions"] = emotion_result

            # organize critters by category
            mentioned_creatures = {cat: [] for cat in self.flora_fauna_data.keys()}

            for category, species_dict in self.flora_fauna_data.items():
                for lemma_val in species_dict:
                    original_val = self.lemma_to_original[lemma_val]
                    if original_val in species_mentions:
                        mention_count = len(species_mentions[original_val])
                        mentioned_creatures[category].extend([original_val] * mention_count)

            # 🦁 The circle of lifeeeeee, and it rules us alllllll 🦁
            predator_count = 0
            prey_count = 0

            for species, mentions in species_mentions.items():
                lemma_sp = self.original_to_lemma.get(species.lower(),
                                                      self.lemmatizer.lemmatize(species.lower()))
                flags = self.species_flags.get(lemma_sp, {})

                if flags.get("predator", False):
                    predator_count += len(mentions)
                if flags.get("prey", False):
                    prey_count += len(mentions)

            return {
                "file_path": file_path,
                "culture": culture,
                "continent": continent,
                "word_count": word_count,
                "creatures": mentioned_creatures,
                "species_mentions": species_mentions,
                "predator_count": predator_count,
                "prey_count": prey_count,
            }

        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            return None

    def process_stories(self, stories_directory: str) -> List[Dict[str, Any]]:
        """
        Processes a batches of stories all at once from a directory.

        Args:
            stories_directory: The folder with all our stories

        Returns:
            A list of all the processed stories
        """
        stories = []
        total_stories = sum(len(files) for _, _, files in os.walk(stories_directory))
        print(f"🌿 Processing {total_stories} stories...")

        # finally processing 🎉
        with tqdm(total=total_stories, desc="🌱 Processing Stories", unit="story") as pbar:
            for root, _, files in os.walk(stories_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, stories_directory)
                    parts = relative_path.split(os.path.sep)

                    continent = parts[0] if len(parts) >= 1 else "Unknown"
                    culture = parts[1] if len(parts) >= 2 else "Unknown"

                    story_data = self.process_story(file_path, culture, continent)
                    if story_data:
                        log_story_to_jsonl(story_data)
                        stories.append(story_data)

                    pbar.update(1)

        print(f"Successfully processed {len(stories)} stories!")
        return stories
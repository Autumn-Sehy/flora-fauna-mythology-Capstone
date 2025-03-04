#!/usr/bin/env python3

# Standard library
import json
import random
from pathlib import Path
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass

# Third party
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Local
from config import Config
from utils import load_data, get_species_mentions  # Import unified utilities


@dataclass
class LabelStudioTask:
    text: str
    labels: List[Dict[str, Any]]
    meta: Dict[str, Any]


class LabelStudioPreparator:
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path("label_studio_data")
        self.output_dir.mkdir(exist_ok=True)
        self.lemmatizer = WordNetLemmatizer()

        print("🌻🌲 Loading flora and fauna 🌲🌻")
        self.flora_fauna_data, self.lemma_to_original, self.original_to_lemma, \
            self.multi_word_terms, self.species_flags = load_data(self.config.FLORA_FAUNA_PATH)

    def get_all_story_paths(self) -> List[Path]:
        """Find all story files in the data directory"""
        stories_dir = Path(self.config.DATA_PATH)
        return list(stories_dir.rglob("*.txt"))

    def process_story_for_labeling(self, story_path: Path) -> Dict:
        """
        Process a story for Label Studio labeling with pre-annotations.
        """
        try:
            with open(story_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                return None

            sentences = sent_tokenize(text)
            indexed_sentences = list(enumerate(sentences))

            species_mentions = get_species_mentions(
                text, sentences, indexed_sentences,
                self.lemma_to_original, self.original_to_lemma,
                self.multi_word_terms, self.lemmatizer,
                output_format="annotation"
            )

            if not species_mentions:
                return None

            return {
                "text": text,
                "labels": species_mentions,
                "corefText": text,
                "meta": {
                    "file_path": str(story_path),
                    "culture": story_path.parent.name,
                    "continent": story_path.parent.parent.name
                }
            }

        except Exception as e:
            print(f"❌ Error processing {story_path}: {str(e)}")
            return None

    def create_label_studio_tasks(self, num_stories: int = 500) -> List[Dict]:
        """
        Create Label Studio tasks from randomly selected stories.
        🐡 Pre-annotates flora and fauna mentions.🪼
        """
        all_story_paths = self.get_all_story_paths()
        selected_paths = random.sample(all_story_paths, min(num_stories, len(all_story_paths)))

        tasks = []

        for story_path in tqdm(selected_paths, desc="🌱 Processing myths🌱"):
            relative_path = story_path.relative_to(self.config.DATA_PATH)
            story_name = story_path.stem
            culture = story_path.parent.name
            continent = story_path.parent.parent.name

            story_data = self.process_story_for_labeling(story_path)
            if story_data and len(story_data['labels']) > 0:
                story_data["meta"].update({
                    "story_name": story_name,
                    "relative_path": str(relative_path),
                    "absolute_path": str(story_path),
                    "culture": culture,
                    "continent": continent
                })

                task = {
                    "data": {
                        "text": story_data["text"],
                        "culture": story_data["meta"]["culture"],
                        "continent": story_data["meta"]["continent"],
                        "story_name": story_data["meta"]["story_name"],
                        "relative_path": story_data["meta"]["relative_path"],
                        "absolute_path": story_data["meta"]["absolute_path"],
                        "corefText": story_data["corefText"],
                    },
                    "predictions": [
                        {
                            "model_version": "flora_v1",
                            "result": story_data["labels"]
                        }
                    ]
                }
                tasks.append(task)

        return tasks

    def prepare_label_config(self) -> str:
        """ 🌼Generate the XML config to make everything very colorful.🌾"""
        labels_list = []
        for category, creatures in self.flora_fauna_data.items():
            category_name = category.replace('_', ' ').title()
            labels_list.append(f'<!-- {category_name} -->')
            for _, original_name in sorted(creatures.items()):
                color = f'#{random.randint(0, 0xFFFFFF):06x}'
                labels_list.append(f'<Label value="{original_name}" background="{color}"/>')

        labels_xml = "\n          ".join(labels_list)

        choices_list = []
        for category, creatures in self.flora_fauna_data.items():
            category_name = category.replace('_', ' ').title()
            choices_list.append(f'<!-- {category_name} -->')
            for _, original_name in sorted(creatures.items()):
                choices_list.append(f'<Choice value="{original_name}"/>')

        choices_xml = "\n          ".join(choices_list)

        return f'''<View>
    <Header value="Story Information"/>
    <Text name="story_info" value="Culture: $culture&#10;Continent: $continent&#10;Story: $story_name&#10;Path: $relative_path"/>

    <Header value="Story Text"/>
    <Text name="text" value="$text"/>

    <!-- Entity Labels -->
    <Labels name="label" toName="text" showInline="true">
          {labels_xml}
          <!-- For pronouns and references -->
          <Label value="Pronoun/Reference" background="#808080"/>
    </Labels>

    <!-- Coreference Relations -->
    <Relations>
        <Relation value="Coreference"/>
    </Relations>

    <!-- Single-choice classification for reference type -->
    <Choices name="reference_type" toName="text" choice="single" showInline="true">
          {choices_xml}
    </Choices>
</View>'''

    def export_for_label_studio(self, tasks: List[Dict]):
        """Export tasks and config for Label Studio"""
        output_path = self.output_dir / "tasks.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        config_path = self.output_dir / "label_config.xml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(self.prepare_label_config())

        print(f"\n🌿 Exported {len(tasks)} pre-annotated stories")


def main():
    parser = argparse.ArgumentParser(description='Prep data')
    parser.add_argument('--num-stories', type=int, default=500,
                        help='Number of stories to randomly select (default: 500)')
    args = parser.parse_args()

    config = Config()
    preparator = LabelStudioPreparator(config)

    print("🌵 Initializing Label Studio task generation...")
    tasks = preparator.create_label_studio_tasks(args.num_stories)
    preparator.export_for_label_studio(tasks)
    print("🍀 Done and ready for annotation!")


if __name__ == "__main__":
    main()
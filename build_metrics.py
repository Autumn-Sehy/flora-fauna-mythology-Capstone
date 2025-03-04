import json
import os
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict, Counter


def shannon_diversity(species_counts):
    """
    This is for calculating Shannon Diversity Index (H').
    According to NIMBioS, " Like Simpson's index,
    Shannon's index accounts for both abundance and evenness of the species present."

    H' = -sum(p_i * ln(p_i))
    where p_i is the proportion of individuals of species i.
    """
    total = sum(species_counts.values())
    if total == 0:
        return 0
    proportions = [count / total for count in species_counts.values() if count > 0]
    return -sum(p * math.log(p) for p in proportions)


def simpson_diversity(species_counts):
    """
    This is for calculating the Simpson diversity index (D).
    According to the Barcelona Field Studies Centre, it is,
    "a measure of diversity which takes into account the number of species present,
    as well as the relative abundance of each species."

    D = 1 - sum(p_i^2)
    where p_i is the proportion of individuals of species i.
    """
    total = sum(species_counts.values())
    if total == 0:
        return 0
    proportions = [count / total for count in species_counts.values()]
    return 1 - sum(p * p for p in proportions)


def calculate_evenness(shannon_index, species_count):
    """
    This is for calculating the evenness index (J).
    According to the Intergovernmental Science-Policy Platform on Biodiversity and Ecosystem Services,
    evenness is "the similarity of abundances of each species in an environment".

    J = H' / ln(S)
    where H' is Shannon diversity index and S is the number of species.
    """
    if species_count <= 1:
        return 1.0
    return shannon_index / math.log(species_count)


def save_metrics_to_json(metrics, filename, output_dir):
    """Save metrics to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    print(f"🦉 Saving metrics to {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"🦊 Successfully saved {filename}")


def load_stories_from_jsonl(jsonl_path):
    """Load all processed stories from JSONL."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"🐍 JSONL file not found: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)

    stories = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="🐢 Loading stories"):
            try:
                stories.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"🦂 Error parsing JSON line: {e}")
    return stories


def compute_metrics_from_jsonl(jsonl_file=None, output_dir="metrics"):
    """Compute multiple metrics and save them to JSON files."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use Config.OUTPUT_JSONL if no file is specified
    if jsonl_file is None:
        try:
            from config import Config
            jsonl_file = Config.OUTPUT_JSONL
        except (ImportError, AttributeError):
            jsonl_file = os.path.join("logs", "processed_stories.jsonl")

    print(f"🦁 Loading stories from {jsonl_file}...")
    stories = load_stories_from_jsonl(jsonl_file)
    print(f"🦓 Calculating metrics for {len(stories)} stories...")
    metrics = {
        # For tracking emotions and sentiments
        "flora_fauna_emotions": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "top_emotions_per_culture": defaultdict(lambda: defaultdict(list)),
        "most_positive_emotions": defaultdict(lambda: defaultdict(list)),
        "most_negative_emotions": defaultdict(lambda: defaultdict(list)),
        "sentiment_by_continent": defaultdict(lambda: defaultdict(list)),

        # For tracking predator/prey relationships
        "predator_prey_ratio": defaultdict(lambda: {"predator_count": 0, "prey_count": 0}),
        "avg_predator_prey_ratio": defaultdict(list),

        # For diversity metrics (ALL mentions)
        "all_species_by_culture": defaultdict(Counter),
        "all_species_by_continent": defaultdict(Counter),
        "all_categories_by_culture": defaultdict(Counter),

        # For diversity metrics (UNIQUE species per story)
        "set_species_by_culture": defaultdict(Counter),
        "set_species_by_continent": defaultdict(Counter),
        "set_categories_by_culture": defaultdict(Counter),

        # Species occurrence tracking
        "species_occurrence": defaultdict(lambda: defaultdict(int))
    }

    # Process each story with a progress bar
    for story in tqdm(stories, desc="🦒 Processing stories"):
        # Use explicit dictionary checks instead of get() for culture and continent
        culture = "Unknown"
        if "culture" in story:
            culture = story["culture"]

        continent = "Unknown"
        if "continent" in story:
            continent = story["continent"]

        if "species_mentions" in story:
            sentiment_data = story["species_mentions"]

            # For unique species count per story
            story_species_set = set()
            story_categories_set = set()

            # Track flora & fauna emotions by category
            if "creatures" in story:
                for category, creatures in story["creatures"].items():
                    # For categories per story
                    story_categories_set.add(category)

                    for creature in creatures:
                        # For species per story
                        story_species_set.add(creature)

                        # Track ALL mentions
                        metrics["all_species_by_culture"][culture][creature] += 1
                        metrics["all_species_by_continent"][continent][creature] += 1
                        metrics["all_categories_by_culture"][culture][category] += 1

                        # Track general occurrence
                        metrics["species_occurrence"][continent][creature] += 1

                        # Track emotions - use dict.get() only for collections that might not exist
                        if creature in sentiment_data:
                            for mention in sentiment_data[creature]:
                                if "emotions" in mention and mention["emotions"]:
                                    metrics["flora_fauna_emotions"][culture][category][creature].append(
                                        mention["emotions"])

                # Add the UNIQUE species and categories from this story
                for species in story_species_set:
                    metrics["set_species_by_culture"][culture][species] += 1
                    metrics["set_species_by_continent"][continent][species] += 1

                for category in story_categories_set:
                    metrics["set_categories_by_culture"][culture][category] += 1

            # Predator/prey count per culture
            # Use explicit checks with default values instead of get()
            predator_count = 0
            if "predator_count" in story:
                predator_count = story["predator_count"]

            prey_count = 0
            if "prey_count" in story:
                prey_count = story["prey_count"]

            metrics["predator_prey_ratio"][culture]["predator_count"] += predator_count
            metrics["predator_prey_ratio"][culture]["prey_count"] += prey_count

            # Predator/prey ratio per continent
            if prey_count > 0:
                metrics["avg_predator_prey_ratio"][continent].append(predator_count / prey_count)

            # Sentiment & emotions by species
            for creature, mentions in sentiment_data.items():
                for mention in mentions:
                    if "sentiment" in mention and mention["sentiment"]:
                        # Use direct access with default where appropriate
                        score = 0
                        if "score" in mention["sentiment"]:
                            score = mention["sentiment"]["score"]

                        metrics["sentiment_by_continent"][continent][creature].append(score)

                        if score > 0:
                            metrics["most_positive_emotions"][culture][creature].append(mention["sentiment"])
                        elif score < 0:
                            metrics["most_negative_emotions"][culture][creature].append(mention["sentiment"])

    # Calculate top emotions per flora/fauna
    for culture, categories in metrics["flora_fauna_emotions"].items():
        for category, creatures in categories.items():
            for creature, emotions in creatures.items():
                emotion_count = defaultdict(int)
                for emotion_data in emotions:
                    if isinstance(emotion_data, dict) and "label" in emotion_data:
                        emotion_count[emotion_data["label"]] += 1
                sorted_emotions = sorted(emotion_count.items(), key=lambda x: x[1], reverse=True)
                metrics["top_emotions_per_culture"][culture][creature] = sorted_emotions[:5]

    # Compute averages for predator/prey ratios per continent
    avg_predator_prey_ratio_dict = {}
    for cont, ratios in metrics["avg_predator_prey_ratio"].items():
        if ratios:
            avg_predator_prey_ratio_dict[cont] = float(np.mean(ratios))
        else:
            avg_predator_prey_ratio_dict[cont] = 0.0

    # Calculate avg sentiment by species & continent
    avg_sentiment_by_continent = defaultdict(dict)
    for continent, creatures in metrics["sentiment_by_continent"].items():
        for creature, scores in creatures.items():
            if scores:
                avg_sentiment_by_continent[continent][creature] = float(np.mean(scores))

    # Sort species by occurrence count
    top_species_by_continent = {}
    for continent, creatures in metrics["all_species_by_continent"].items():
        sorted_creatures = sorted(creatures.items(), key=lambda x: x[1], reverse=True)
        top_species_by_continent[continent] = sorted_creatures[:20]  # Top 20 species

    # Calculate predator/prey ratio for each culture
    calculated_predator_prey_ratio = {}
    for culture, counts in metrics["predator_prey_ratio"].items():
        pred_count = counts["predator_count"]
        prey_count = counts["prey_count"]
        if prey_count > 0:
            calculated_predator_prey_ratio[culture] = pred_count / prey_count
        else:
            calculated_predator_prey_ratio[culture] = pred_count if pred_count > 0 else 0

    print("🦅 Calculating diversity metrics...")

    # metricsmetricsmetrics
    def calc_diversity(counts_dict):
        if not counts_dict:
            return {}

        shannon_index = shannon_diversity(counts_dict)
        simpson_index = simpson_diversity(counts_dict)
        evenness = calculate_evenness(shannon_index, len(counts_dict))

        return {
            "shannon_diversity": shannon_index,
            "simpson_diversity": simpson_index,
            "evenness": evenness,
            "count": len(counts_dict),
            "total": sum(counts_dict.values())
        }

    # gotta get them diversity metrics yo
    diversity_results = {
        "all_culture_diversity": {culture: calc_diversity(counts) for culture, counts in
                                  metrics["all_species_by_culture"].items()},
        "all_continent_diversity": {continent: calc_diversity(counts) for continent, counts in
                                    metrics["all_species_by_continent"].items()},
        "all_category_diversity": {culture: {**calc_diversity(counts), "category_distribution": dict(counts)}
                                   for culture, counts in metrics["all_categories_by_culture"].items()},
        "set_culture_diversity": {culture: calc_diversity(counts) for culture, counts in
                                  metrics["set_species_by_culture"].items()},
        "set_continent_diversity": {continent: calc_diversity(counts) for continent, counts in
                                    metrics["set_species_by_continent"].items()},
        "set_category_diversity": {culture: {**calc_diversity(counts), "category_distribution": dict(counts)}
                                   for culture, counts in metrics["set_categories_by_culture"].items()}
    }

    # dump all this stuff to json
    metrics_to_save = {
        "flora_fauna_emotions.json": metrics["flora_fauna_emotions"],
        "top_emotions_per_culture.json": metrics["top_emotions_per_culture"],
        "predator_prey_ratio.json": metrics["predator_prey_ratio"],
        "calculated_predator_prey_ratio.json": calculated_predator_prey_ratio,
        "avg_predator_prey_ratio.json": avg_predator_prey_ratio_dict,
        "most_positive_emotions.json": metrics["most_positive_emotions"],
        "most_negative_emotions.json": metrics["most_negative_emotions"],
        "species_occurrence.json": metrics["species_occurrence"],
        "avg_sentiment_by_continent.json": avg_sentiment_by_continent,
        "top_species_by_continent.json": top_species_by_continent,

        # ALL mentions diversity metrics
        "all_culture_diversity.json": diversity_results["all_culture_diversity"],
        "all_continent_diversity.json": diversity_results["all_continent_diversity"],
        "all_category_diversity.json": diversity_results["all_category_diversity"],

        # UNIQUE species set diversity metrics
        "set_culture_diversity.json": diversity_results["set_culture_diversity"],
        "set_continent_diversity.json": diversity_results["set_continent_diversity"],
        "set_category_diversity.json": diversity_results["set_category_diversity"]
    }

    # Save all metrics
    for filename, data in metrics_to_save.items():
        save_metrics_to_json(data, filename, output_dir)

    print(f"🦄 Successfully computed and saved all metrics from {len(stories)} stories!")
    return output_dir
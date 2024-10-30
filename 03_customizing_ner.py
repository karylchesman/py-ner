import spacy
import json
import random


file = "data/hp_training_data.json"


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Starting iteration: " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text], [annotations], drop=0.2, sgd=optimizer, losses=losses
                )
                print(losses)
    return nlp


TRAIN_DATA = load_data(file)
nlp = train_spacy(TRAIN_DATA, 30)
nlp.to_disk("hp_ner_model")

import json

with open("data/hp.txt", "r", encoding="utf-8") as f:
    text = f.read().split("\n\n")

characters_names = []
with open("data/hp_characters.json", "r", encoding="utf-8") as f:
    characters = json.load(f)
    for character in characters:
        names = character.split()
        for name in names:
            if name not in ("and", "the", "The"):
                name = name.replace(",", "").strip()
                characters_names.append(name)

for segment in text:
    segment = segment.strip()
    segment = segment.replace("\n", " ")
    print(segment)

    punctuation = """!()-[]{};:'"\,<>./?@#$%^&*_~"""

    for ele in segment:
        if ele in punctuation:
            segment = segment.replace(ele, "")

    words = segment.split()
    i = 0
    for word in words:
        if word in characters_names:
            if words[i - 1][0].isupper():
                print(f"Found Character(s): {words[i-1]} {word}")
            else:
                print(f"Found Character(s): {word}")

        i = i + 1

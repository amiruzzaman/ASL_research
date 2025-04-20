from ml.api.english_to_gloss import EnglishToGloss


model = EnglishToGloss()

while True:
    print("Enter Sentence to translate to ASL Gloss:")
    sentence = input().lower()

    print(model.translate(sentence))
    print()
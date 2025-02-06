from dataprocess import get_data


def main():
    data, gloss_id, text_id = get_data()

    # Fetches a sample from the batch and prints the gloss and token
    train_glosses, train_text = next(iter(data))
    gloss = train_glosses[0]
    text = train_text[0] 

    print(f"Gloss: {[gloss_id[val.item()] for val in gloss]}")
    print(f"Gloss Tokens: {gloss}\n")
    print(f"Text: {[text_id[val.item()] for val in text]}")
    print(f"Text Tokens: {text}\n\n")


if __name__ == "__main__":
    main()
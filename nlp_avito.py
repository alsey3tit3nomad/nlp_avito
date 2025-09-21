import pandas as pd
import argparse
from transformers import AutoTokenizer


def run_pipeline(file_path, encoding):
    raw_tokens = []
    if (encoding is None):
        encoding = "utf-8-sig"
    with open(file_path, 'r', encoding=encoding) as f:
        raw_tokens = f.read().split()

    df = []
    for string in raw_tokens[1:]:
        index_comma = string.find(',')
        idx = string[:index_comma]
        text = string[index_comma + 1:]
        df.append({'id' : int(idx), 'text' : text})

    df = pd.DataFrame(data = df, columns=['id', 'text'])
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    def get_space_positions(text):
        text = text.lower()
        tokens = tokenizer.tokenize(text)
        positions = []
        current_pos = 0
        for token in tokens[:-1]:
            current_pos += len(token)
            positions.append(current_pos)
        return str(positions)

    df["predicted_positions"] = df["text"].apply(get_space_positions)
    df.to_csv('submission.csv', columns=['id', 'predicted_positions'], index=False)

def main():
    parser = argparse.ArgumentParser(description='Токенизация текста')
    parser.add_argument("--file", required=True, help='Путь к файлу .txt')
    parser.add_argument("--encoding", help="Кодировка", required=False)
    args = parser.parse_args()
    run_pipeline(args.file, args.encoding)

if __name__ == '__main__':
    main()

import argparse
import json
from pathlib import Path
import string

def normalize(text: str):
    result = normalize_text(text)
    print(result["status"])
    if result["status"] not in [0,1]:
        print(f"Normalization did not work 100% '{result["input_text"]}'.")

    return result["normalized_text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=str, help="Directory of the dataset.")
    parser.add_argument("--out_dir", required=False, type=str, help="Output directory.")
    parser.add_argument("--punctuation", default=True, type=bool, help="Clean punctuation with Pythons string module.")
    parser.add_argument("--normalize", default=False, type=bool, help="Normalize with Slovene_normalizator.")
    args = parser.parse_args()

    if args.normalize:
        # https://github.com/clarinsi/Slovene_normalizator
        from normalizator.main_normalization import normalize_text
        import nltk
        nltk.download('punkt_tab')

    directory = args.dir
    input_files = ['all.nemo', 'dev.nemo', 'test.nemo', 'train.nemo']

    for input_file in input_files:
        print(f"Processing {input_file}...")
        with open(f'{directory}/{input_file}', 'r', encoding='utf-8') as infile:
            if args.out_dir:
                out_dir = args.out_dir
            else:
                out_dir = f'{directory}/clean'
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_file = f'{out_dir}/{input_file}'
            with open(out_file, 'w', encoding='utf-8') as outfile:
                for idx, line in enumerate(infile):
                    entry = json.loads(line)
                    print(f"{idx} - {entry["text"]}")
                    if args.punctuation:
                        entry['text'] = entry['text'].translate(str.maketrans('', '', string.punctuation))
                    if args.normalize:
                        entry['text'] = normalize(entry["text"])
                    outfile.write(
                        json.dumps(entry, ensure_ascii=False)
                        + '\n'
                    )

            print(f"Processed file saved as {out_file}")
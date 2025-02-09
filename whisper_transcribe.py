import argparse
import json

import pandas as pd
import torch
import whisper
from tqdm import tqdm

from metrics_calculator import MetricsCalculator

# def get_top_n_hypotheses(results: list[DecodingResult], n: int):
#     top_n_hypotheses: list[str] = []
#     for result in results:
#         if len(top_n_hypotheses) < n and len(result.text) > 0 and result not in top_n_hypotheses:
#             top_n_hypotheses.append(result.text)
#     if len(top_n_hypotheses) < n:
#         for _ in range(n - len(top_n_hypotheses)):
#             random_hypotheses_to_use_again = copy.deepcopy(random.choice(top_n_hypotheses))
#             top_n_hypotheses.append(random_hypotheses_to_use_again)
#     return top_n_hypotheses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create hypotheses-to-transcript mapping dataset with Whisper')
    parser.add_argument('--manifest_file_path', type=str, help='Path to the NeMo manifest file')
    parser.add_argument('--beams_file_path', type=str, help='Path to output the resulting beams')
    parser.add_argument('--beam_width', type=int, help='Width of the resulting beams')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(name="base", device=device)
    # normalizer = BasicTextNormalizer()
    calc = MetricsCalculator()

    count = 0
    wer = 0
    df = pd.DataFrame(columns=["hypotheses", "asr_scores"])
    with open(args.manifest_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            manifest_entry = json.loads(line)
            audio = whisper.load_audio(manifest_entry["audio_filepath"])
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            options = whisper.DecodingOptions(language="sl", beam_size=args.beam_width)
            results = whisper.decode(model, mel, options)

            hypotheses = [result.text for result in results]
            asr_scores = [result.avg_logprob for result in results]

            df_to_add = pd.DataFrame({"hypotheses": hypotheses, "asr_scores": asr_scores})

            df = pd.concat([df, df_to_add], ignore_index=True)

            count += 1
            current_wer = calc.calculate_wer([hypotheses[0]], [manifest_entry["text"]])
            wer += current_wer

    print(f'WER = {wer/count}')
    df.to_csv(args.beams_file_path, sep='\t', index=False, header=False)


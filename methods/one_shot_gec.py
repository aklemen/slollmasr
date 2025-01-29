from torch_datasets.hypotheses_dataset import HypothesesDataset
from utils.build_chats import build_chats
from utils.examples import examples
from utils.prompter import Prompter


class OneShotGec:
    def __init__(self, llm_name: str, tokenizer_name: str, batch_size: int = 8):
        self.prompter = Prompter(llm_name, tokenizer_name, batch_size)

    def run(self, dataset: HypothesesDataset):
        chats = build_chats(dataset, self._build_chat)
        return self.prompter.execute_chats(chats)

    def _build_chat(self, hypotheses: list[str]) -> list[dict[str, str]]:
        beam_size = len(hypotheses)
        return [{
            "role": "user",
            "content": (
                f"Izvedi popravljanje napak na najboljših {beam_size} izhodih, ki jih je generiral sistem za samodejno razpoznavanje govora (ASR). "
                f"Hipoteze so navedene po vrstnem redu glede na njihovo posteriorno verjetnost iz sistema ASR. "
                f"Prosim, izpiši le popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed. "
                f"Tukaj je primer naloge:\n\n"
                f"{self._stringify_hypotheses(examples[0]['hypotheses'][:beam_size])}\n\n"
                f"Tvoj izhod: {examples[0]['ground_truth']}\n\n"
                f"Prosim, zgleduj se po zgornjem primeru. Prosim, začni:\n\n"
                f"{self._stringify_hypotheses(hypotheses)}"
            )
        }]

    def _stringify_hypotheses(self, hypotheses: list[str]) -> str:
        return "\n".join([
            f"<hipoteza{idx+1}> {hypothesis} </hipoteza{idx+1}>" for idx, hypothesis in enumerate(hypotheses)
        ])

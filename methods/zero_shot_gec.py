from torch_datasets.hypotheses_dataset import HypothesesDataset
from utils.build_chats import build_chats
from utils.prompter import Prompter


class ZeroShotGec:
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
                f"Hipoteze, navedene po vrstnem redu glede na njihovo posteriorno verjetnost iz sistema ASR, so naslednje:\n\n"
                f"{self._stringify_hypotheses(hypotheses)}\n\n"
                "Prosim, izpiši le popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed."
            )
        }]

    def _stringify_hypotheses(self, hypotheses: list[str]) -> str:
        return "\n".join([
            f"<hipoteza{idx}> {hypothesis} </hipoteza{idx}>" for idx, hypothesis in enumerate(hypotheses)
        ])

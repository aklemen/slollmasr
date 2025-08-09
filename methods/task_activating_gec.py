from torch_datasets.hypotheses_dataset import HypothesesDataset
from utils.build_chats import build_prompt_dataset
from utils.prompter import Prompter
from utils.examples import examples


class TaskActivatingGec:
    def __init__(self, llm_name: str, tokenizer_name: str, batch_size: int = 8):
        self.prompter = Prompter(llm_name, tokenizer_name, batch_size)

    def run(self, dataset: HypothesesDataset):
        chats = build_prompt_dataset(dataset, self._build_chat)
        return self.prompter.execute_chats(chats)

    def _build_chat(self, hypotheses: list[str]) -> list[dict[str, str]]:
        beam_size = len(hypotheses)
        return [
            {
                "role": "user",
                "content": "Ali veš, kaj je razpoznavanje govora?"
            },
            {
                "role": "assistant",
                "content":
                    "Da, vem, kaj je razpoznavanje govora. Razpoznavanje govora, znano tudi kot avtomatsko razpoznavanje govora "
                    "(angl. Automatic Speech Recognition) je postopek pretvorbe govorjenega jezika v besedilo. "
                    "Ta tehnologija vključuje uporabo algoritmov in modelov strojnega učenja za analizo in transkripcijo akustičnih "
                    "značilk izgovorjenih besed in povedi. Razpoznavanje govora se uporablja na različnih področjih, "
                    "kot so govorni pomočniki, avtomatizirani telefonski sistemi in storitve transkripcije."
            },
            {
                "role": "user",
                "content": "Ali veš, kaj je ponovno ocenjevanje (angl. rescoring) z jezikovnim modelom v razpoznavanju govora?"
            },
            {
                "role": "assistant",
                "content":
                    "Da, vem, kaj je ponovno ocenjevanje z jezikovnim modelom v razpoznavanju govora. "
                    "Ponovno ocenjevanje z jezikovnim modelom je tehnika, ki se uporablja za izboljšanje natančnosti sistemov za "
                    "razpoznavanje govora. Vključuje uporabo ločenega jezikovnega modela za oceno verjetnosti danega seznama hipotez. "
                    "Ta ločeni model je običajno bolj kompleksen in zmogljiv kot osnovni jezikovni model, uporabljen za transkripcijo, "
                    "in se uporablja za ponovno ocenjevanje transkripta na podlagi verjetnosti, da se besede pojavijo v danem "
                    "kontekstu. Postopek ponovnega ocenjevanja vključuje posredovanje izhoda začetnega jezikovnega modela, ki "
                    "običajno temelji na statističnih metodah, kot so skriti Markovovi modeli, bolj naprednemu jezikovnemu modelu, "
                    "kot je na primer jezikovni model, ki temelji na nevronskih mrežah, da se ustvari natančnejši transkript. "
                    "To dosežemo s ponovnim razvrščanjem danih hipotez na podlagi verjetnosti, podanih s strani naprednejšega "
                    "jezikovnega modela. Ponovno ocenjevanje z jezikovnim modelom dokazano izboljša natančnost sistemov za "
                    "razpoznavanje govora, zlasti v hrupnih ali zahtevnih okoljih, kjer začetni jezikovni model morda ne deluje dobro.",
            },
            {
                "role": "user",
                "content": f"Ali lahko podaš primer ponovnega ocenjevanja {beam_size} najboljših hipotez z jezikovnim modelom?",
            },
            {
                "role": "assistant",
                "content":
                    f"Seveda, tukaj je primer ponovnega ocenjavanja {beam_size} najboljših hipotez z jezikovnim modelom:\n"
                    f"{self._stringify_hypotheses(examples[1]['hypotheses'][:beam_size])}\n"
                    f"Po ponovnem ocenjevanju mislim, da bi moral pravilni transkript tega govora biti: {examples[1]['ground_truth']} ",
            },
            {
                "role": "user",
                "content":
                    f"Odlično si se odrezal! Zdaj ti bom dal en resničen primer. {beam_size} najboljših hipotez je:\n"
                    f"{self._stringify_hypotheses(examples[0]['hypotheses'][:beam_size])}\n"
                    f"Pričakujem, da je tvoj izhod: {examples[0]['ground_truth']}\n"
                    f"Z uporabo tega primera prosim podaj pravilni transkript za naslednjih {beam_size} hipotez:\n"
                    f"{self._stringify_hypotheses(hypotheses)}\n"
                    "Prosim, izpiši le transkript, brez dodatnih razlag ali besed."
            }
        ]

    def _stringify_hypotheses(self, hypotheses: list[str]) -> str:
        return '\n'.join([
            f"   {idx+1}. {hypothesis}" for idx, hypothesis in enumerate(hypotheses)
        ])

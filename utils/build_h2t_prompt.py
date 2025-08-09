prompt_template = ("### Navodilo:\n"
                   "Spodaj je najboljša hipoteza, ki jo je za avdio posnetek generiral sistem za razpoznavanje govora. "
                   "Preglej jo in jo s pomočjo ostalih hipotez popravi, če je potebno. "
                   "Potem izpiši končni transkript.\n\n"
                   "### Najboljša hipoteza:\n{best_hypothesis}\n\n"
                   "### Ostale hipoteze:\n{other_hypotheses}\n\n"
                   f"### Transkript:\n")

def build_h2t_prompt(hypotheses: list[str]):
    return prompt_template.format_map({
        "best_hypothesis": hypotheses[0],
        "other_hypotheses": "\n".join(hypotheses[1:]),
    })
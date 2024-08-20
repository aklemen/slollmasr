import logging

from HypothesesDataset import HypothesesDataset


class BestHypothesesSelector:

    @staticmethod
    def select(dataset: HypothesesDataset, custom_scores: list[float] = None):
        hypotheses = dataset.get_hypotheses_texts()
        scores = dataset.get_hypotheses_scores()

        if custom_scores is not None:
            logging.info("Custom scores were provided - using custom scores.")
            scores = custom_scores

        if len(scores) != len(hypotheses):
            raise Exception('Scores should have the same length as the number of hypotheses')

        beam_size = dataset.get_beam_size()
        best_hypotheses = []
        print(len(hypotheses))
        print(beam_size)
        for i in range(int(len(hypotheses) / beam_size)):
            hypotheses_for_sample = hypotheses[i*5:i*5 + 5]
            scores_for_sample = scores[i*5:i*5 + 5]
            max_value = max(scores_for_sample)
            max_index = scores_for_sample.index(max_value)
            best_hypotheses.append(hypotheses_for_sample[max_index])

        return best_hypotheses

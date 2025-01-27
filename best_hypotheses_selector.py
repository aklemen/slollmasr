from torch_datasets.hypotheses_dataset import HypothesesDataset


class BestHypothesesSelector:
    @staticmethod
    def select(dataset: HypothesesDataset, custom_scores: list[float] = None):
        hypotheses = dataset.get_hypotheses_texts()
        scores = dataset.get_hypotheses_scores()

        if custom_scores is not None:
            scores = custom_scores

        if len(scores) != len(hypotheses):
            raise Exception('Scores should have the same length as the number of hypotheses')

        beam_size = dataset.get_beam_size()
        num_samples = dataset.get_num_of_samples()
        best_hypotheses = []
        best_scores = []
        best_indices = []
        for i in range(num_samples):
            start_idx = i * beam_size
            end_idx = (i + 1) * beam_size
            hypotheses_for_sample = hypotheses[start_idx:end_idx]
            scores_for_sample = scores[start_idx:end_idx]
            max_value = max(scores_for_sample)
            max_index = scores_for_sample.index(max_value)
            best_hypotheses.append(hypotheses_for_sample[max_index])
            best_scores.append(scores_for_sample[max_index])
            best_indices.append(start_idx + max_index)

        return best_hypotheses, best_scores, best_indices

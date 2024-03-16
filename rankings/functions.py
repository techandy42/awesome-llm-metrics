from typing import Dict, List

def rank_models_by_evaluations(evaluations: Dict[str, List[float]], weights: Dict[str, float]) -> List[int]:
  # Initialize a list to store the weighted sum of ranks for each index (model)
  weighted_sum_of_ranks = [0] * len(evaluations[next(iter(evaluations))])

  # For each metric, rank the models based on their scores, apply the weight, and sum these ranks for each model
  for metric, scores in evaluations.items():
      # Rank the models for the current metric (1 is the highest rank)
      ranks = [sorted(scores, reverse=True).index(score) + 1 for score in scores]
      # Apply the weight for the current metric and add to the total sum of ranks for each model
      weighted_ranks = [rank * weights[metric] for rank in ranks]
      weighted_sum_of_ranks = [sum_rank + weighted_rank for sum_rank, weighted_rank in zip(weighted_sum_of_ranks, weighted_ranks)]

  print(weighted_sum_of_ranks)

  # Sort the weighted sums to identify their order
  sorted_ranks = sorted(set(weighted_sum_of_ranks))

  # Map each unique sum to its rank
  rank_map = {rank: i + 1 for i, rank in enumerate(sorted_ranks)}

  # Assign ranks based on the weighted sums (with ties receiving the same rank)
  final_ranks = [rank_map[sum_rank] for sum_rank in weighted_sum_of_ranks]

  return final_ranks

if __name__ == "__main__":
  # Example Models
  models = [
         {
            "source": "cohere",
            "model": "command-r",
         },
         {
            "source": "openai",
            "model": "gpt-4-0125-preview",
         },
         {
            "source": "groq",
            "model": "mixtral-8x7b-32768",
         },
      ]
  # Example Values
  evaluations = {
    'bleu': [0.35, 0.5, 0.65],
    'rouge1': [0.6, 0.7, 0.45],
    'rouge2': [0.35, 0.3, 0.25],
    'rougel': [0.4, 0.5, 0.3],
    'rougelsum': [-0.1, 0.2, -0.05],
  }
  weights = {
    'bleu': 1.0,
    'rouge1': 0.25,
    'rouge2': 0.25,
    'rougel': 0.25,
    'rougelsum': 0.25,
  }

  model_ranks = rank_models_by_evaluations(evaluations=evaluations, weights=weights)

  for i in range(len(models)):
    models[i]['rank'] = model_ranks[i]

  models.sort(key=lambda x: x['rank'])

  print("Model Ranks")
  for model in models:
    print(f"Model: {model['source']}, Model Name: {model['model']}, Rank: {model['rank']}")

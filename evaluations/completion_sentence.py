from models.openai_module import OpenAIModel
from models.anthropic_module import AnthropicModel
from models.cohere_module import CohereModel
from models.groq_module import GroqModel
from models.google_generative_ai_module import GoogleGenerativeAIModel
from models.google_vertex_ai_module import GoogleVertexAIModel
from inferences.functions import complete_sentence
from metrics.base_metrics import calc_rouge_score
from typing import List, Dict, Tuple

def evaluate_completion_sentence(prompts: List[str], references: List[List[str]], model_details: List[Dict[str, str]], evaluation_details: List[str]) -> Tuple[List[List[str]], Dict[str, List[float]]]:
  models = []

  for model_detail in model_details:
    if model_detail["source"] == "openai":
      models.append(OpenAIModel(model=model_detail["model"]))
    elif model_detail["source"] == "anthropic":
      models.append(AnthropicModel(model=model_detail["model"]))
    elif model_detail["source"] == "cohere":
      models.append(CohereModel(model=model_detail["model"]))
    elif model_detail["source"] == "groq":
      models.append(GroqModel(model=model_detail["model"]))
    elif model_detail["source"] == "genai":
      models.append(GoogleGenerativeAIModel(model=model_detail["model"]))
    elif model_detail["source"] == "vertexai":
      models.append(GoogleVertexAIModel(model=model_detail["model"]))
    
  all_results = complete_sentence(
    prompts=prompts,
    models=models
  )

  models_completions = [[None for _ in prompts] for _ in models]

  for prompt_results in all_results:
    for prompt_index, model_index, completion in prompt_results:
      models_completions[model_index][prompt_index] = completion

  evaluations = {}

  if 'rouge' in evaluation_details:
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeLsum_scores = []

    for model_completions in models_completions:
      rouge_score = calc_rouge_score(model_completions, references)
      rouge1_scores.append(rouge_score['rouge1'])
      rouge2_scores.append(rouge_score['rouge2'])
      rougeL_scores.append(rouge_score['rougeL'])
      rougeLsum_scores.append(rouge_score['rougeLsum'])

    evaluations['rouge1'] = rouge1_scores
    evaluations['rouge2'] = rouge2_scores
    evaluations['rougel'] = rougeL_scores
    evaluations['rougelsum'] = rougeLsum_scores

  return models_completions, evaluations

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Functions Test Code Arguments")
  parser.add_argument('--all', action='store_true', help='Include All Models')
  parser.add_argument('--openai', action='store_true', help='Include OpenAI Model')
  parser.add_argument('--anthropic', action='store_true', help='Include Anthropic Model')
  parser.add_argument('--cohere', action='store_true', help='Include Cohere Model')
  parser.add_argument('--groq', action='store_true', help='Include Groq Model')
  parser.add_argument('--genai', action='store_true', help='Include Google Generative AI Model')
  parser.add_argument('--vertexai', action='store_true', help='Include Google Vertex AI Model')
  args = parser.parse_args()

  prompts = ["Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then", 
             "A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. the pans"]
  references = [
    [", the man adds wax to the windshield and cuts it.", ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.", ", the man puts on a christmas coat, knitted with netting.", ", the man continues removing the snow on his car."],
    ["contain egg yolks and baking soda.", "are then sprinkled with brown sugar.", "are placed in a strainer on the counter.", "are filled with pastries and loaded into the oven."],
  ]
  model_details = []

  if args.all:
      models = [
         {
            "source": "openai",
            "model": "gpt-4-0125-preview",
         },
         {
            "source": "anthropic",
            "model": "claude-3-opus-20240229",
         },
         {
            "source": "cohere",
            "model": "command-r",
         },
         {
            "source": "groq",
            "model": "mixtral-8x7b-32768",
         },
         {
            "source": "genai",
            "model": "gemini-1.0-pro",
         },
         {
            "source": "vertexai",
            "model": "gemini-1.0-pro",
         },
      ]
  else:
      if args.openai:
          model_details.append({
            "source": "openai",
            "model": "gpt-4-0125-preview",
         })
      if args.anthropic:
          model_details.append({
            "source": "anthropic",
            "model": "claude-3-opus-20240229",
         })
      if args.cohere:
          model_details.append({
            "source": "cohere",
            "model": "command-r",
         })
      if args.groq:
          model_details.append({
            "source": "groq",
            "model": "mixtral-8x7b-32768",
         })
      if args.genai:
          model_details.append({
            "source": "genai",
            "model": "gemini-1.0-pro",
         })
      if args.vertexai:
          model_details.append({
            "source": "vertexai",
            "model": "gemini-1.0-pro",
         })
  evaluation_details = ['rouge']

  models_completions, evaluations = evaluate_completion_sentence(prompts, references, model_details, evaluation_details)

  for model_detail, model_completions, rouge1_score, rouge2_score, rougeL_score, rougeLsum_score in zip(model_details, models_completions, evaluations['rouge1'], evaluations['rouge2'], evaluations['rougel'], evaluations['rougelsum']):
    print(f"Model: {model_detail['source']}, Model Name: {model_detail['model']}")
    print()
    for prompt, completion, reference_group in zip(prompts, model_completions, references):
      newline = "\n"
      print(f"Original: {prompt}\nCompletion: {completion}\nCombined: {prompt + ' ' + completion}\nReferences:\n{newline.join(reference_group)}")
      print()
    print("ROUGE-1 Score:")
    print(rouge1_score)
    print("ROUGE-2 Score:")
    print(rouge2_score)
    print("ROUGE-L Score:")
    print(rougeL_score)
    print("ROUGE-Lsum Score:")
    print(rougeLsum_score)
    print()
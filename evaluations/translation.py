from models.openai_module import OpenAIModel
from models.anthropic_module import AnthropicModel
from models.cohere_module import CohereModel
from models.groq_module import GroqModel
from models.google_generative_ai_module import GoogleGenerativeAIModel
from models.google_vertex_ai_module import GoogleVertexAIModel
from inferences.functions import translate
from metrics.base_metrics import calc_bleu_score, calc_rouge_score
from typing import List, Dict, Tuple

def evaluate_translation(prompts: List[str], src_langs: List[str], tgt_langs: List[str], references: List[List[str]], model_details: List[Dict[str, str]], evaluation_details: List[str]) -> Tuple[List[List[str]], Dict[str, List[float]]]:
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
    
  all_results = translate(
    prompts=prompts,
    src_langs=src_langs,
    tgt_langs=tgt_langs,
    models=models
  )

  models_translations = [[None for _ in prompts] for _ in models]

  for prompt_results in all_results:
    for prompt_index, model_index, translation in prompt_results:
      models_translations[model_index][prompt_index] = translation

  evaluations = {}

  if 'bleu' in evaluation_details:
    bleu_scores = []

    for model_translations in models_translations:
      bleu_score = calc_bleu_score(model_translations, references)
      bleu_scores.append(bleu_score['bleu'])

    evaluations['bleu'] = bleu_scores

  if 'rouge' in evaluation_details:
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeLsum_scores = []

    for model_translations in models_translations:
      rouge_score = calc_rouge_score(model_translations, references)
      rouge1_scores.append(rouge_score['rouge1'])
      rouge2_scores.append(rouge_score['rouge2'])
      rougeL_scores.append(rouge_score['rougeL'])
      rougeLsum_scores.append(rouge_score['rougeLsum'])

    evaluations['rouge1'] = rouge1_scores
    evaluations['rouge2'] = rouge2_scores
    evaluations['rougel'] = rougeL_scores
    evaluations['rougelsum'] = rougeLsum_scores

  return models_translations, evaluations

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

  prompts = ["Je suis un etudiant.", "J'aime creme de glace."]
  src_langs = ["French", "French"]
  tgt_langs = ["English", "English"]
  references = [
    ["I am a student.", "I go to the school."],
    ["I like ice cream.", "I enjoy ice cream.", "I love ice cream."],
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
  evaluation_details = ['bleu', 'rouge']

  models_translations, evaluations = evaluate_translation(prompts, src_langs, tgt_langs, references, model_details, evaluation_details)

  for model_detail, model_translations, bleu_score, rouge1_score, rouge2_score, rougeL_score, rougeLsum_score in zip(model_details, models_translations, evaluations['bleu'], evaluations['rouge1'], evaluations['rouge2'], evaluations['rougel'], evaluations['rougelsum']):
    print(f"Model: {model_detail['source']}, Model Name: {model_detail['model']}")
    print()
    for prompt, translation, reference_group in zip(prompts, model_translations, references):
      newline = "\n"
      print(f"Prompt: {prompt}\nTranslation: {translation}\nReferences:\n{newline.join(reference_group)}")
      print()
    print("BLEU Score:")
    print(bleu_score)
    print("ROUGE-1 Score:")
    print(rouge1_score)
    print("ROUGE-2 Score:")
    print(rouge2_score)
    print("ROUGE-L Score:")
    print(rougeL_score)
    print("ROUGE-Lsum Score:")
    print(rougeLsum_score)
    print()

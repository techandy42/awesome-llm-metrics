from models.openai_module import OpenAIModel
from models.anthropic_module import AnthropicModel
from models.cohere_module import CohereModel
from models.groq_module import GroqModel
from models.google_generative_ai_module import GoogleGenerativeAIModel
from models.google_vertex_ai_module import GoogleVertexAIModel
from inferences.functions import complete_missing_word
from metrics.base_metrics import calc_missing_words_accuracy
from typing import List, Dict, Tuple

def evaluate_completion_missing_word(prompts: List[str], references: List[List[str]], answers: List[str], model_details: List[Dict[str, str]], evaluation_details: List[str]) -> Tuple[List[List[str]], Dict[str, List[float]]]:
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
    
  all_results = complete_missing_word(
    prompts=prompts,
    models=models,
    missing_words=references,
  )

  models_completions = [[None for _ in prompts] for _ in models]

  for prompt_results in all_results:
    for prompt_index, model_index, completion in prompt_results:
      models_completions[model_index][prompt_index] = completion

  evaluations = {}

  if 'accuracy' in evaluation_details:
    missing_words_accuracies = []
    for model_completions in models_completions:
      missing_words_accuracy = calc_missing_words_accuracy(model_completions, references, answers)
      missing_words_accuracies.append(missing_words_accuracy['accuracy'])
    evaluations['accuracy'] = missing_words_accuracies

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

  prompts = ["John moved the couch from the garage to the backyard to create space. The _ is small.", 
             "The doctor diagnosed Justin with bipolar and Robert with anxiety. _ had terrible nerves recently.",
             "Dennis drew up a business proposal to present to Logan because _ wants his investment.",
             "Felicia unexpectedly made fried eggs for breakfast in the morning for Katrina and now _ owes a favor.",
             "My shampoo did not lather easily on my Afro hair because the _ is too dirty.",]
  references = [
    ["garage", "backyard"],
    ["Justin", "Robert"],
    ["Dennis", "Logan"],
    ["Felicia", "Katrina"],
    ["shampoo", "hair"],
    ]
  answers = [1, 2, 1, 2, 2]
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
  evaluation_details = ['accuracy']

  models_completions, evaluations = evaluate_completion_missing_word(prompts, references, answers, model_details, evaluation_details)

  for model_detail, model_completions, missing_word_accuracy in zip(model_details, models_completions, evaluations['accuracy']):
    print(f"Model: {model_detail['source']}, Model Name: {model_detail['model']}")
    print()
    for prompt, completion, reference_group in zip(prompts, model_completions, references):
      newline = "\n"
      print(f"Original: {prompt}\nCompletion: {completion}\nCombined: {prompt.replace('_', completion)}\nReferences:\n{newline.join(reference_group)}")
      print()
    print("Missing Word Accuracy:")
    print(missing_word_accuracy)
    print()
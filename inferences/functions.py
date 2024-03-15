from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.base_module import BaseModel

def translate(prompts: List[str], src_langs: List[str], tgt_langs: List[str], models: List[BaseModel]) -> List[List[Tuple[int, int, str]]]:
    all_results = []  # To store results for all prompts

    # Validate input lengths
    if not (len(prompts) == len(src_langs) == len(tgt_langs)):
        raise ValueError("The lengths of prompts, src_langs, and tgt_langs must be equal.")
    
    for prompt_index, (prompt, src_lang, tgt_lang) in enumerate(zip(prompts, src_langs, tgt_langs)):
        with ThreadPoolExecutor() as executor:
            # Mapping each future to a tuple containing model index and prompt index
            future_to_indexes = {
                executor.submit(model.translate, prompt, src_lang, tgt_lang): (model_index, prompt_index)
                for model_index, model in enumerate(models)
            }
            
            temp_results = []
            for future in as_completed(future_to_indexes):
                model_index, prompt_index = future_to_indexes[future]
                translation = future.result()
                # Each result now also includes the prompt index
                temp_results.append((prompt_index, model_index, translation))

            # Sort the results primarily by prompt index, then by model index
            temp_results.sort(key=lambda x: (x[0], x[1]))
            all_results.append(temp_results)
    
    return all_results

def summarize(prompts: List[str], models: List[BaseModel]) -> List[List[Tuple[int, int, str]]]:
    all_results = []

    with ThreadPoolExecutor() as executor:
        for prompt_index, prompt in enumerate(prompts):
            future_to_indexes = {
                executor.submit(model.summarize, prompt): (model_index, prompt_index)
                for model_index, model in enumerate(models)
            }
            
            temp_results = []
            for future in as_completed(future_to_indexes):
                model_index, prompt_index = future_to_indexes[future]
                summary = future.result()
                temp_results.append((prompt_index, model_index, summary))

            temp_results.sort(key=lambda x: (x[0], x[1]))
            all_results.append(temp_results)

    return all_results

def q_and_a(prompts: List[str], models: List[BaseModel]) -> List[List[Tuple[int, int, str]]]:
    all_results = []

    with ThreadPoolExecutor() as executor:
        for prompt_index, prompt in enumerate(prompts):
            future_to_indexes = {
                executor.submit(model.q_and_a, prompt): (model_index, prompt_index)
                for model_index, model in enumerate(models)
            }
            
            temp_results = []
            for future in as_completed(future_to_indexes):
                model_index, prompt_index = future_to_indexes[future]
                answer = future.result()
                temp_results.append((prompt_index, model_index, answer))

            temp_results.sort(key=lambda x: (x[0], x[1]))
            all_results.append(temp_results)

    return all_results

def complete_sentence(prompts: List[str], models: List[BaseModel]) -> List[List[Tuple[int, int, str]]]:
    all_results = []

    with ThreadPoolExecutor() as executor:
        for prompt_index, prompt in enumerate(prompts):
            future_to_indexes = {
                executor.submit(model.complete_sentence, prompt): (model_index, prompt_index)
                for model_index, model in enumerate(models)
            }
            
            temp_results = []
            for future in as_completed(future_to_indexes):
                model_index, prompt_index = future_to_indexes[future]
                completion = future.result()
                temp_results.append((prompt_index, model_index, completion))

            temp_results.sort(key=lambda x: (x[0], x[1]))
            all_results.append(temp_results)

    return all_results

def complete_missing_word(prompts: List[str], models: List[BaseModel], missing_words: List[List[str]]) -> List[List[Tuple[int, int, str]]]:
    all_results = []

    if not (len(prompts) == len(missing_words)):
        raise ValueError("The lengths of prompts and missing_words_list must be equal.")
    
    for prompt_index, (prompt, missing_word_group) in enumerate(zip(prompts, missing_words)):
        with ThreadPoolExecutor() as executor:
            future_to_indexes = {
                executor.submit(model.complete_missing_word, prompt, missing_word_group): (model_index, prompt_index)
                for model_index, model in enumerate(models)
            }
            
            temp_results = []
            for future in as_completed(future_to_indexes):
                model_index, prompt_index = future_to_indexes[future]
                completion = future.result()
                temp_results.append((prompt_index, model_index, completion))

            temp_results.sort(key=lambda x: (x[0], x[1]))
            all_results.append(temp_results)
    
    return all_results

# Test Case
if __name__ == "__main__":
  import argparse
  from models.openai_module import OpenAIModel
  from models.anthropic_module import AnthropicModel
  from models.cohere_module import CohereModel
  from models.groq_module import GroqModel
  from models.google_generative_ai_module import GoogleGenerativeAIModel
  from models.google_vertex_ai_module import GoogleVertexAIModel

  parser = argparse.ArgumentParser(description="Functions Test Code Arguments")
  parser.add_argument('--all', action='store_true', help='Include All Models')
  parser.add_argument('--openai', action='store_true', help='Include OpenAI Model')
  parser.add_argument('--anthropic', action='store_true', help='Include Anthropic Model')
  parser.add_argument('--cohere', action='store_true', help='Include Cohere Model')
  parser.add_argument('--groq', action='store_true', help='Include Groq Model')
  parser.add_argument('--genai', action='store_true', help='Include Google Generative AI Model')
  parser.add_argument('--vertexai', action='store_true', help='Include Google Vertex AI Model')
  args = parser.parse_args()

  models = []

  if args.all:
      models = [OpenAIModel(), AnthropicModel(), CohereModel(), GroqModel(), GoogleGenerativeAIModel(), GoogleVertexAIModel()]
  else:
      if args.openai:
          models.append(OpenAIModel())
      if args.anthropic:
          models.append(AnthropicModel())
      if args.cohere:
          models.append(CohereModel())
      if args.groq:
          models.append(GroqModel())
      if args.genai:
          models.append(GoogleGenerativeAIModel())
      if args.vertexai:
          models.append(GoogleVertexAIModel())

  prompts = ["Berlin is the capital of Germany.", "Barcelona is a city in Spain.", "What is the answer to life, universe, and everything?"]
  src_langs = ["English", "English", "English"]
  tgt_langs = ["German", "Spanish", "French"]
  all_results = translate(prompts, src_langs, tgt_langs, models)
  for prompt_results in all_results:
    for prompt_index, model_index, translation in prompt_results:
      print(models[model_index])
      print(f"===== Translation =====\n{src_langs[prompt_index]}: {prompts[prompt_index]}\n{tgt_langs[prompt_index]}: {translation}\n")

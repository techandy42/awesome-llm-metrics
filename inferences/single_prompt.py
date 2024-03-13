from models.base_module import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

def translate(prompt: str, src_lang: str, tgt_lang: str, models: List[BaseModel]) -> Tuple[List[str], List[int]]:
    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(model.translate, prompt, src_lang, tgt_lang): i for i, model in enumerate(models)}
        
        results = []
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            translation = future.result()
            results.append((index, translation))
    
    # Sort results by model index
    results.sort(key=lambda x: x[0])
    
    return results

def summarize(prompt: str, models: List[BaseModel]) -> Tuple[List[str], List[int]]:
    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(model.summarize, prompt): i for i, model in enumerate(models)}
        
        results = []
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            summary = future.result()
            results.append((index, summary))
    
    results.sort(key=lambda x: x[0])
    
    return results

def q_and_a(prompt: str, models: List[BaseModel]) -> Tuple[List[str], List[int]]:
    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(model.q_and_a, prompt): i for i, model in enumerate(models)}
        
        results = []
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            answer = future.result()
            results.append((index, answer))
    
    results.sort(key=lambda x: x[0])
    
    return results

def complete_sentence(prompt: str, models: List[BaseModel]) -> Tuple[List[str], List[int]]:
    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(model.complete_sentence, prompt): i for i, model in enumerate(models)}
        
        results = []
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            completion = future.result()
            results.append((index, completion))
    
    results.sort(key=lambda x: x[0])
    
    return results

def complete_missing_word(prompt: str, models: List[BaseModel]) -> Tuple[List[str], List[int]]:
    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(model.complete_missing_word, prompt): i for i, model in enumerate(models)}
        
        results = []
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            completion = future.result()
            results.append((index, completion))
    
    results.sort(key=lambda x: x[0])
    
    return results

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

    # Translation
    src_lang = "English"
    tgt_lang = "French"
    prompt = "What is the answer to life, universe, and everything?"
    results = translate(prompt, src_lang, tgt_lang, models)
    for index, translation in results:
        print(models[index])
        print(f"===== Translation =====\nEnglish: {prompt}\nFrench: {translation}\n")

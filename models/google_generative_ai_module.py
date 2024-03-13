import os
from dotenv import load_dotenv
import google.generativeai as genai
from models.base_module import BaseModel

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

class GoogleGenerativeAIModel(BaseModel):
  def __init__(self, api_key: str = GOOGLE_API_KEY, model: str = "gemini-1.0-pro"):
    genai.configure(api_key=api_key)
    self.client = genai.GenerativeModel(model)
    self.model = model

  def call(self, prompt: str) -> str:
    response = self.client.generate_content(prompt)
    return response.text

  def __str__(self) -> str:
    return f"GoogleGenerativeAI,{self.model}"

  def translate(self, prompt: str, src_lang: str, tgt_lang: str) -> str:
    prompt_template = f"""Instruction:
- Translate the following text from {src_lang} to {tgt_lang}.
- Do not output anything else.

Text:
{prompt}
"""
    
    return self.call(prompt=prompt_template)

  def summarize(self, prompt: str) -> str:
    prompt_template = f"""Instruction:
- Summarize the following text.
- Do not output anything else.

Text:
{prompt}
"""
    
    return self.call(prompt=prompt_template)

  def q_and_a(self, prompt: str) -> str:
    prompt_template = f"""Instruction:
- Provide a definitive answer to the following question.
- Do not output anything else.

Question:
{prompt}
"""
    
    return self.call(prompt=prompt_template)

  def complete_sentence(self, prompt: str) -> str:
    prompt_template = f"""Instruction:
- Complete the following sentence.
- Only output the completion.

Sentence:
{prompt}
"""

    return self.call(prompt=prompt_template)

  def complete_missing_word(self, prompt: str) -> str:
    prompt_template = f"""Instruction:
- Output the missing word from the following sentence.
- Only output the missing word.

Sentence:
{prompt}
"""

    return self.call(prompt=prompt_template)
  
# Test Cases
if __name__ == "__main__":
  gemini_model = GoogleGenerativeAIModel()

  # Model Details
  print(gemini_model)

  # Translation
  src_lang = "English"
  tgt_lang = "French"
  prompt = "What is the answer to life, universe, and everything?"
  response = gemini_model.translate(prompt=prompt, src_lang=src_lang, tgt_lang=tgt_lang)
  print(f"===== Translation =====\nEnglish: {prompt}\nFrench: {response}\n")

  # Summarization
  prompt = "The Alexander the Great was a king of the ancient Greek kingdom of Macedon and a member of the Argead dynasty. He was born in Pella in 356 BC and succeeded his father Philip II to the throne at the age of 20. He spent most of his ruling years on an unprecedented military campaign through Western Asia and Northeastern Africa, and by the age of thirty, he had created one of the largest empires of the ancient world, stretching from Greece to northwestern India. He was undefeated in battle and is widely considered one of history's most successful military commanders."
  response = gemini_model.summarize(prompt=prompt)
  print(f"===== Summarization =====\nOriginal: {prompt}\nSummary: {response}\n")

  # Q&A
  prompt = "What happens to you if you eat watermelon seeds?"
  response = gemini_model.q_and_a(prompt=prompt)
  print(f"===== Q&A =====\nQuestion: {prompt}\nAnswer: {response}\n")

  # Completion - Sentence
  prompt = "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then"
  response = gemini_model.complete_sentence(prompt=prompt)
  print(f"===== Completion - Sentence =====\nPrompt: {prompt}\nCompletion: {response}\nCombined: {prompt + ' ' + response}\n")

  # Completion - Missing Word
  prompt = "John moved the couch from the garage to the backyard to create space. The _ is small."
  response = gemini_model.complete_missing_word(prompt=prompt)
  print(f"===== Completion - Missing Word =====\nPrompt: {prompt}\nMissing Word: {response}\nCombined: {prompt.replace('_', response)}\n")

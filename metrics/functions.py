from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from typing import List, Dict, Tuple
from pprint import pprint

def calc_bleu(translation: str, references: List[str]) -> Tuple[float, float, float, float]:
  tokenized_translation = translation.split()
  tokenized_references = [reference.split() for reference in references]

  return (
    sentence_bleu(tokenized_references, tokenized_translation, weights=(1, 0, 0, 0)),
    sentence_bleu(tokenized_references, tokenized_translation, weights=(0, 1, 0, 0)),
    sentence_bleu(tokenized_references, tokenized_translation, weights=(0, 0, 1, 0)),
    sentence_bleu(tokenized_references, tokenized_translation, weights=(0, 0, 0, 1)),
  )

def calc_rouge(output: str, reference: str) -> Dict[str, Dict[str, float]]:
  rouge = Rouge()
  return rouge.get_scores(output, reference)[0]

def calc_rouge_mult(outputs: List[str], references: List[str]) -> List[Dict[str, Dict[str, float]]]:
  rouge = Rouge()
  return rouge.get_scores(outputs, references)

# Test Cases
if __name__ == "__main__":
  newline = "\n"
  translation = "the cat is on the mat"
  references = ["there is a cat on the mat", "a cat is on the mat"]
  bleu_scores = calc_bleu(translation, references)
  print(f"===== Translation BLEU Scores =====\nTranslation: {translation}\nReferences:\n{newline.join(references)}\nUnigram: {bleu_scores[0]}\nBigram: {bleu_scores[1]}\nTrigram: {bleu_scores[2]}\nQuadgram: {bleu_scores[3]}")

  output = "the cat is on the mat"
  reference = "there is a cat on the mat"
  rouge_scores = calc_rouge(output, reference)
  print(f"===== Translation ROUGE Scores =====\nOutput: {output}\nReference: {reference}\nROUGE Scores:")
  pprint(rouge_scores)

  outputs = [
    "the cat is on the mat",
    "the dog is on the porch",
  ]
  references = [
    "there is a cat on the mat",
    "there is a dog on the porch",
  ]
  rouge_scores = calc_rouge_mult(outputs, references)
  print(f"===== Translation ROUGE Scores (Multiple) =====\nOutputs:\n{newline.join(outputs)}\nReferences:\n{newline.join(references)}\nROUGE Scores:")
  pprint(rouge_scores)

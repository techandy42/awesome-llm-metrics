import evaluate
from typing import List, Dict
from pprint import pprint

bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')

def calc_bleu_score(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
  return bleu.compute(predictions=predictions, references=references, max_order = 2)

def calc_rouge_score(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
  return rouge.compute(predictions=predictions, references=references)

def calc_missing_words_accuracy(predictions: List[str], references: List[List[str]], answers: List[int]) -> Dict[str, float]:
    correct_count = 0
    
    for pred, ref, ans in zip(predictions, references, answers):
        if pred == ref[ans - 1]:
            correct_count += 1
            
    accuracy = correct_count / len(predictions)
    return {
       'accuracy': accuracy
    }

if __name__ == "__main__":
  predictions = ["Transformers Transformers are fast plus efficient", 
                "Good Morning", "I am waiting for new Transformers"]
  references = [
                ["HuggingFace Transformers are quick, efficient and awesome", 
                "Transformers are awesome because they are fast to execute"], 
                ["Good Morning Transformers", "Morning Transformers"], 
                ["People are eagerly waiting for new Transformer models", 
                "People are very excited about new Transformers"]
  ]
  missing_words_predictions = ['a', 'c']
  missing_words_references = [['a', 'b'], ['c', 'd']]
  missing_words_answers = [1, 2]

  bleu_score = calc_bleu_score(predictions, references)
  rouge_score = calc_rouge_score(predictions, references)
  missing_words_accuracy = calc_missing_words_accuracy(missing_words_predictions, missing_words_references, missing_words_answers)

  print("Predictions:")
  pprint(predictions)
  print()
  print("References:")
  pprint(references)
  print()
  print("BLEU Scores:")
  pprint(bleu_score)
  print()
  print("ROUGE Scores:")
  pprint(rouge_score)
  print()
  print("Missing Words Accuracy:")
  pprint(missing_words_accuracy)

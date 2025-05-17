import numpy as np
import evaluate
from transformers import GenerationConfig

#Prints total number of parameters available and the number of tunable parameters.
def trainable_parameters_stats(model):
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")


# Generate text using the pretrained model
# You can play with max_new_tokens and other parameters
def generate_model_output(model, tokenizer, inputs, max_new_tokens=50):
    output = model.generate(
        input_ids=inputs["input_ids"], 
        generation_config=GenerationConfig(max_new_tokens=max_new_tokens)
    )
    # Decode the generated tokens and remove special tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text


#Compute ROUGE score between predictions and references.
class RougeEvaluator:
    def __init__(self):
        self.rouge = evaluate.load('rouge')
    
    def compute_rouge(self, predictions, references):
        return self.rouge.compute(predictions=predictions, references=references, use_aggregator=True, use_stemmer=True,)

def rouge_comparison(model_results, reference_results):
    # Calculate the difference between model results and reference results
    improvement = (np.array(list(model_results.values())) - np.array(list(reference_results.values())))
    
    # Print the improvement in percentage for each ROUGE metric
    for key, value in zip(model_results.keys(), improvement):
        print(f'{key}: {value*100:.2f}%')
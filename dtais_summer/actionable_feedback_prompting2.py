import json
import os
import tempfile
import random
import requests
import statistics


#system prompt

generic_prompt = f"""You are an expert academic reviewer that lists actionable feedback for the author of the content that is provided in the user prompt. Present your feedback as a list of bullet points.
"""

#, such as "Clarify X", "Provide evidence for Y", or "Reorganize Z" for example.
#specific_prompt = f"""You are an expert academic reviewer that lists actionable feedback for the author of the content that is provided in the user prompt. This means your feedback can be immediately implemented by the author to improve the paper. Frame each point as a specific, concrete suggestion or revision command."""

specific_prompt = """You are an expert academic writing assistant that lists actionable feedback for the author of the content that is provided in the user prompt.
This means your feedback can be immediately implemented by the author to improve the paper. 

Actionable feedback is:
- **Specific**: Points to a particular section, sentence, or idea in the paper.
- **Constructive**: Suggests improvements without being vague.
- **Feasible**: Offers suggestions the author can realistically implement.
- **Context-aware**: Understands the paper’s topic, genre, and purpose.
- **Improvement-oriented**: Aims to make the paper more clear, persuasive, or well-structured.

Do NOT just give praise or high-level commentary. Instead, focus on concrete guidance the author can follow to revise their work.
Present your feedback as a list of bullet points.

"""
def build_prompt(paper):
  metadata = paper.get('metadata') #metadata dictionary that contains the actual contents of the paper
  content_list = metadata.get('sections')
  #print(content_list)
  #print(content_list[4].get('text'))
  #print(type(content_list[4]))

  paper_content = str(metadata.get('sections'))
  prompt = f"""Paper content: {paper_content}
  Provide your suggestions for the author to improve the paper. """

  return prompt


def model_forecasting(model, system_prompt, prompt):
    #print(prompt)
    # Send request to Ollama
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model, #llama3.2:3b , "qwen3:latest"
            "system": system_prompt,
            "prompt": prompt, 
            "stream": False
            }
    )
    result = res.json()
    return result

def give_feedback(pdf_path, system_prompt, results):
    with open(pdf_path, 'r') as f1:
        paper = json.load(f1) #json file contents for one research paper

    prompt = build_prompt(paper)
    model = "llama3:8b"
    output = model_forecasting(model, system_prompt, prompt)
    json_response = output["response"]
    #print(json_response)
    results[paper.get("name")] = {
        "review": json_response
    }
    return results


def generic_sample(dir_path, sample_size, output_path):
    paper_names = os.listdir(dir_path)
    results = {}
    for i in range(0, sample_size): 
        pdf_path = os.path.join(dir_path, paper_names[i])
        results = give_feedback(pdf_path, generic_prompt, results)
    
    with open(output_path,'a') as f3:
        json.dump(results,f3)


def specific_sample(dir_path, sample_size, output_path):
    paper_names = os.listdir(dir_path)
    results = {}
    for i in range(0, sample_size): 
        pdf_path = os.path.join(dir_path, paper_names[i])
        results = give_feedback(pdf_path, specific_prompt, results)
    
    with open(output_path,'a') as f3:
        json.dump(results,f3)



dir_path = "/GSEHD/home/g34371231/Desktop/Alice/PeerRead/data/iclr_2017/train/parsed_pdfs"
generic_output_path = "/GSEHD/home/g34371231/Desktop/Alice/PeerRead/dtais_summer/generic_iclr_100.json"
specific_output_path = "/GSEHD/home/g34371231/Desktop/Alice/PeerRead/dtais_summer/specific_iclr_100_2.json"
sample_size = 100

#generic_sample(dir_path, sample_size, generic_output_path)
specific_sample(dir_path, sample_size, specific_output_path)
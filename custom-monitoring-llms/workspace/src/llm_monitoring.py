
import boto3
import pathlib
import os
import re
import base64
import json

from langchain.llms import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.utils import enforce_stop_tokens

RELEVANCE_TEMPLATE = """\n\nHuman: Evaluate if the Answer is relevant to the Question. answer 1 if it is relevant. answer 0 if it is relevant.\n\nAssistant:I will only answer 1 or 0.
\nHuman:
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?
Answer:\nThe PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India 
\nAssistant: 1
\nHuman:
Question: {question}
Answer: {answer}
\nAssistant:
""" 

EVALUATOR = PromptTemplate(template=RELEVANCE_TEMPLATE, input_variables=["question", "answer"])

# Evaluate relevance between questions and answers
def evaluate_relevance(questions, answers, llm):
    score = 0
    
    llm_chain = LLMChain(llm=llm, prompt=EVALUATOR)

    for q, a in zip(questions, answers):
        
        try:
            text = llm_chain.run(question=q, answer=a).strip()
            text = enforce_stop_tokens(text, ["\n\nHuman:"])
            print(f"\n-----------\n")
            print(f"* {q} --> {a} --> <<{text}>>")
            relevant = int(text)
        except ValueError:
            relevant = 0
            
        score += relevant

    return score/ len(questions) * 100

def base64_to_string(base64_string):
    base64_bytes = base64_string.encode('ascii')
    string_bytes = base64.b64decode(base64_bytes) 
    return string_bytes.decode('utf-8')

def extract_instructions(text):
    pattern = r"### Instruction\n(.*?)\n\n"
    match = re.search(pattern, text)
    return match.group(1)

def extract_answers(text):
    pattern = r"### Answer\n\n(.*)|### Answer\n(.*)"
    match = re.search(pattern, text)

    return match.group(1) or match.group(2)   

# Helper function to extract question and answer from dataset
def extract_qa(input_data, output_data):
    question = extract_instructions(json.loads(input_data)["text"])
    
    generated_text = json.loads(base64_to_string(output_data))["outputs"][0]["generated_text"]
    answer = extract_answers(generated_text)
    return question, answer

def main():    
    
    # Load dataset
    questions, answers = [], []
    infer_dir = os.environ['dataset_source']

    for filepath in pathlib.Path(infer_dir).rglob('*.jsonl'):

        with open(filepath.absolute(), 'r') as f:
            for line in f:
                jsonl = json.loads(line)
                input_data = jsonl['captureData']['endpointInput']['data']
                output_data = jsonl['captureData']['endpointOutput']['data']
                
                q, a = extract_qa (input_data, output_data)
                questions.append(q)
                answers.append(a)

    print(questions)
    print(answers)
    
    # Initialize LLMs            
    llm = Bedrock(
        model_id="anthropic.claude-v2",
        model_kwargs={"max_tokens_to_sample": 200,
                    "temperature": 0},
        client=boto3.client("bedrock-runtime", region_name='us-west-2'),
    )
    
    # Evaluate relevance
    score = evaluate_relevance(questions, answers, llm)
    print(f"relevancy score: {score} /100")

    # Save results
    output = {
        'relevancy': score/100,
#         'end_time': os.environ['end_time']
    }

    with open(f"{os.environ['output_path']}/results.json", 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    
    main()

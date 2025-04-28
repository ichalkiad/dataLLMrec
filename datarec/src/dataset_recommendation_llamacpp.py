from prompt_templates import ChatPromptTemplate
from llama_cpp import Llama
import argparse
import time
import jsonlines
import numpy as np

# Use "embed" method to extract output embedding
# Check stopping criteria for generation

def get_full_prompt(dataset_name=None, dataset_author=None, dataset_description=None):

    template_messages = [
            {"role": "system", "content": "You are a helpful and harmless AI assistant with expertise on dataset curation and developmet. "
                                          "You will be provided with a textual description of a dataset."
                                          "Your task is to analyze the textual description of the dataset and suggest tasks that it can help to solve."
                                          "Respond in English with an itemized list of tasks.\n"
                                          "\n"
                                          "\n"
                                          "**Example start:**\n"
                                        "**Input:**\n"
                                        "\n"
                                        "Dataset: MATH-5000, OpenAI, 'This dataset contains a subset of 500 mathematics problems from the MATH benchmark that OpenAI created in their Let's Verify Step by Step paper. Text generation. Text'\n"
                                        "**Output:**\n"                        
                                        "Task 1. This dataset is good for text generation problems that involve mathematical expressions.\n"
                                        "Task 2. The data can be used to improve the reasoning capabilities of large language models."
                                        "**Example end.**\n"
                                        "**Now, please analyze the following dataset description.\n"\
                                        "**User Query:**\n"                                        
                                        },
            {"role": "user", "content": "Dataset name: {{name}}. Dataset author: {{author}}. Dataset description: {{datasetcard}}. Provide an itemized list of tasks that the dataset can solve."}
    ]
    template_variables = ["name", "author", "datasetcard"]
    metadata = {
         "name": "Task recommender",
         "description": "A chat prompt for recommending tasks that suit a dataset with given characteristics.",
         "tags": [],
         "version": "0.0.9",
         "author": ""
    }
    prompt_template = ChatPromptTemplate(
        template=template_messages,
        template_variables=template_variables,
        metadata=metadata
    )
    # print(prompt_template)
    prompt = prompt_template.populate(
        name=dataset_name,
        author=dataset_author,
        datasetcard=dataset_description
    )

    return prompt



def main(dir_out):
    parser = argparse.ArgumentParser(description="Test top-K and top-P sampling with Phi-3-mini")
    parser.add_argument("--model", type=str, default="phi-3-mini-4k-instruct.Q4_K_M.gguf", help="Path to GGUF model file")
    parser.add_argument("--datasetname", type=str, default=None, help="Dataset name.")
    parser.add_argument("--datasetauthor", type=str, default=None, help="Dataset author.")
    parser.add_argument("--datasetcard", type=str, default=None, help="Dataset card.")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (nucleus) sampling parameter")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=2000, help="Maximum tokens to generate")
    parser.add_argument("--n_threads", type=int, default=4, help="Number of CPU threads to use")
    parser.add_argument("--seed", type=int, default=254, help="Random seed for reproducibility")
    args = parser.parse_args()

    prompt = None                          
    print(f"Loading model: {args.model}")
    print(f"Sampling params: top_k={args.top_k}, top_p={args.top_p}, temp={args.temp}")
    
    # prompt = "You are a helpful and harmless AI assistant with expertise on dataset curation and developmet."\
    #          "You will be provided with a textual description of a dataset."\
    #          "Your task is to analyze the textual description of the dataset and suggest tasks that it can help to solve."\
    #          "Respond in English with an itemized list of tasks."\
    #          "\n"\
    #          "\n"\
    #          "**Example:**\n"\
    #          "**Input:**\n"\
    #          "\n"\
    #          "Dataset: MATH-5000, OpenAI, 'This dataset contains a subset of 500 mathematics problems from the MATH benchmark that OpenAI created in their Let's Verify Step by Step paper. Text generation. Text'\n"\
    #          "**Output:**\n"\
    #          "Task 1. This dataset is good for text generation problems that involve mathematical expressions.\n"\
    #          "Task 2. The data can be used to improve the reasoning capabilities of large language models.\n"\
    #          "**Now, analyze the following dataset description.\n"\
    #          "**User Query:**\n"\
    #          "{}".format(args.dataprompt)

    if args.datasetname is None:
        args.datasetname = "Llama-Nemotron-Post-Training-Dataset-v1"
    if args.datasetauthor is None:
        args.datasetauthor = "Nvidia"    
    if args.datasetcard is None:
        args.datasetcard = "This dataset is a compilation of SFT and RL data that supports improvements of math, code, general reasoning, and instruction following capabilities of the original Llama instruct model, in support of NVIDIA’s release of Llama-3.3-Nemotron-Super-49B-v1 and Llama-3.1-Nemotron-Nano-8B-v1." \
                        "Llama-3.3-Nemotron-Super-49B-v1 is a large language model (LLM) which is a derivative of Meta’s Llama-3.3-70B-Instruct (AKA the reference model). Llama-3.1-Nemotron-Nano-8B-v1 is a large language model (LLM) which is a derivative of Meta Llama-3.1-8B-Instruct (AKA the reference model). They are aligned for human chat preferences, and tasks." \
                        "These models offer a great tradeoff between model accuracy and efficiency. Efficiency (throughput) directly translates to savings. Using a novel Neural Architecture Search (NAS) approach, we greatly reduce the model’s memory footprint, enabling larger workloads, as well as fitting the model on a single GPU at high workloads (H100-80GB). This NAS approach enables the selection of a desired point in the accuracy-efficiency tradeoff. The model supports a context length of 128K." \
                        "This dataset release represents a significant move forward in openness and transparency in model development and improvement. By releasing the complete training set, in addition to the training technique, tools and final model weights, NVIDIA supports both the re-creation and the improvement of our approach."
    prompt = get_full_prompt(dataset_name=args.datasetname, dataset_author=args.datasetauthor, dataset_description=args.datasetcard)

    # Load the model
    start_time = time.time()
    llm = Llama(
        model_path=args.model,
        n_ctx=2048,
        n_threads=args.n_threads,
        seed=args.seed
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    llm_embed = Llama(
        model_path=args.model,
        n_ctx=2048,
        n_batch=512,
        n_threads=2,
        seed=args.seed,
        embedding=True
    )

    # Generate completion
    start_time = time.time()
    # output = llm.create_completion(
    #     prompt,
    #     max_tokens=args.max_tokens,
    #     top_k=args.top_k,
    #     top_p=args.top_p,
    #     temperature=args.temp,
    # )
    output = llm.create_chat_completion(
        prompt,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temp,
    )
    gen_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*50)
    print(f"PROMPT: {prompt}")
    # print(f"COMPLETION: {output['choices'][0]['text']}")
    reponse = output['choices'][0]["message"]["content"]
    print("RESPONSE: {}".format(reponse))
    print("="*50)
    print(f"Generation time: {gen_time:.2f} seconds")
    print(f"Speed: {args.max_tokens / gen_time:.2f} tokens/sec")

    dataset_embedding = llm_embed.embed(prompt[1]["content"], normalize=False, truncate=False)
    dataset_embedding_vec_avg = np.mean(np.array(dataset_embedding), axis=0).tolist()

    task_list = reponse.split("\n")
    with jsonlines.open("{}/task_embeddings.jsonl".format(dir_out), "a") as jsonl_write:
        for i in range(len(task_list)):
            task = task_list[i]
            try:
                task_vec = llm_embed.embed(task, normalize=False, truncate=False)
                task_vec_avg = np.mean(np.array(task_vec), axis=0).tolist()
                jsonl_write.write({"dataset": args.datasetname, "dataset_author": args.datasetauthor, 
                                "dataset_embedding": dataset_embedding_vec_avg, "avg_task_{}_embedding".format(i): task_vec_avg})
            except Exception as e:
                print(f"Error during embedding: {e}")
                

if __name__ == "__main__":
    dir_out = "/tmp/"
    main(dir_out)
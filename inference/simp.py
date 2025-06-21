# run_multi_gpu.py

from vllm import LLM, SamplingParams
import time

def main():
    # --- 1. Set the model and tensor parallel size ---
    # This model is a good size for demonstrating parallelism.
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # This is the key parameter. It tells vLLM to shard the model across 4 GPUs.
    TENSOR_PARALLEL_SIZE = 4 

    print(f"Initializing LLM '{model_id}' with tensor_parallel_size={TENSOR_PARALLEL_SIZE}...")
    
    # --- 2. Initialize the LLM with Tensor Parallelism ---
    # vLLM will automatically handle the distribution of the model across the GPUs.
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=8192,  # Important to manage memory, as discussed
    )
    print("LLM Initialized. All GPUs are loaded.")

    # --- 3. Define Prompts and Sampling Parameters ---
    prompts = [
        "What are the main advantages of using a multi-GPU setup for deep learning?",
        "Write a python function that calculates the fibonacci sequence.",
        "Explain the concept of tensor parallelism in simple terms.",
        "Tell me a short story about four spaceships exploring a new galaxy together.",
    ]
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=512)

    # --- 4. Generate Text ---
    print("\n--- Generating Responses ---")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    print(f"Generation took {end_time - start_time:.2f} seconds.")

    # --- 5. Print the Outputs ---
    print("\n--- Model Outputs ---")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}\n")

if __name__ == '__main__':
    main()
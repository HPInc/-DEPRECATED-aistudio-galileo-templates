#import tensorrt_llm



# To do tests:
# * Create a better prompt
# * Verify Prompt as a list of {Role: info} information
# * Verify StopWords

from tensorrt_llm import LLM, SamplingParams

prompt_template = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for travel tips and recommendations. You prefer to keep the answers short, clear and concise.<|eot_id|><|start_header_id|>user<|end_header_id|>

What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''

import os
if __name__ == '__main__':
    os.environ["HF_HOME"] = "/home/jovyan/local/hugging_face"
    os.environ["HF_HUB_CACHE"] = "/home/jovyan/local/hugging_face/hub"

    
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=100)
    llm = LLM(model="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1")
    
    outputs = llm.generate([prompt_template], sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

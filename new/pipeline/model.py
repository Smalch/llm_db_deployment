from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import torch
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
device = 0 if torch.cuda.is_available() else -1  # set to GPU if available
print(f"Using device: {device}", torch.cuda.is_available())
device = 'auto'

class Model:
    def __init__(self, model_name='falcon-40b-instruct', max_length=2000, temperature=0.9):
        model_name = 'adw_llm'
        self.max_length = max_length
        self.temperature = temperature

        #model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True)
        if 'mpt-' in model_name:
            self._llm = self.mpt(model_name, max_length, temperature)
        else:
            model_path = f'models/{model_name}'
            self._llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True, torch_dtype=torch.bfloat16)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, return_tensor='pt', device_map=device)

    @property
    def llmmpt(self):
        return self._llm

    def llm(self, max_length=2000, temperature=0.9):
        pipe = pipeline(
            "text-generation",
            model=self._llm,
            tokenizer=self.tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        return HuggingFacePipeline(pipeline=pipe)
        #return self.pipe

    def pipe_only(self, max_length=2000, temperature=0.9):
        pipe = pipeline(
            "text-generation",
            model=self._llm,
            tokenizer=self.tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        return pipe

    # def model_start(self, model_path):
    #     return AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True)




    def mpt(self, model_name, max_length, temperature):

        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_id in stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        model_path = f'models/{model_name}'
        config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.max_seq_len = 83968 # (input + output) tokens can now be up to 83968
        #config.attn_config['attn_impl'] = 'triton'
        config.init_device = 'cuda:0'
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


        stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            max_length=83968,
            stopping_criteria=stopping_criteria,
            top_k=0,
            top_p=0.15,
            repetition_penalty=1.1,
            trust_remote_code=True,
            max_new_tokens=max_length,
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        return HuggingFacePipeline(pipeline=pipe)


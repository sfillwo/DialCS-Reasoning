from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import torch
import ezpyz as ez

@dataclass
class DataConfig:
    format: Optional[str] = None
    context_length: Optional[int] = None
    prefix: Optional[str] = None
    parse_func: Optional[callable] = None
    inference_representation: Optional[dict[str, str]] = None
    fewshots: Optional[dict[str, str]] = None

@dataclass
class GenerationConfig:
    repetition_penalty: Optional[float] = None
    num_beams: Optional[int] = None
    num_beam_groups: Optional[int] = None
    diversity_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    use_commonsense: bool = False

class ModelWrapper:

    def __init__(self,
                 name: str,
                 modelpath: str,
                 use_commonsense: bool,
                 generation_config: GenerationConfig,
                 data_config: DataConfig,
                 batch_size: int,
                 num_to_gen: int,
                 quantization_config: dict = None,
                 device: str = 'cpu',
                 store: ez.filelike|None = 'llm_cache',
                 inference_preprocessor: Optional[callable] = None,
                 speaker_tag='Speaker (Other):',
                 listener_tag='Listener (You):'
    ):
        self.name = name
        self.modelpath = modelpath
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_commonsense = use_commonsense
        self.model = None
        self.tokenizer, self.collator = None, None
        self.generation_config = generation_config
        self.data_config = data_config
        self.quantization_config = quantization_config
        self.batch_size = batch_size
        self.num_to_gen = num_to_gen
        self.store = store
        self.inference_preprocessor = inference_preprocessor
        self.speaker_tag = speaker_tag
        self.listener_tag = listener_tag

    def load_model(self, checkpoint):
        print(f"Loading {checkpoint}: model...")
        return None, None, None

    def attach(self):
        if self.quantization_config is None:
            self.model = self.model.to(self.device)

    def unattach(self):
        if self.quantization_config is None:
            self.model = self.model.to('cpu')

    def clear_cuda_cache(self):
        del self.model
        torch.cuda.empty_cache()

    def format_data(self, data):
        ...

    def generate(self, *args, **kwargs):
        ...

    def parse_response(self, data):
        ...
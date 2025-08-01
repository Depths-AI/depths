from typing import Optional, Dict, List, Any, Tuple

class LLMLogsConfig:
    '''
    Instructions on how to log LLM calls
    '''
    def __init__(
        self,
        store_input_text: Optional[bool]=False,
        store_output_text: Optional[bool]=False
        ):

        self.store_input_text=store_input_text
        self.store_output_text=store_output_text

class DepthsLogger:
    '''
    Core logger class
    '''
    def __init__(
        self, 
        llm_logging_config: Optional[LLMLogsConfig]=LLMLogsConfig()
        ):


        self.llm_logging_config=llm_logging_config
# titans/scripts/utils/interactive_demo.py
import torch
import gradio as gr
from typing import Dict, List, Optional,Tuple

from inference import predictor
from ...inference import TitansPredictor
from ...utils.config import ConfigLoader
from ...utils.logging import setup_logger

class TitansDemo:
    def __init__(
        self,
        predictor: TitansPredictor,
        max_length: int = 8192
    ):
        self.predictor = predictor
        self.max_length = max_length
        self.memory_state = None
        
    def generate(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float,
        use_memory: bool
    ) -> Tuple[str, Dict[str, float]]:
        """Generate text continuation."""
        memory = self.memory_state if use_memory else None
        generated_text, new_memory = self.predictor.predict(
            text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            memory_state=memory
        )
        
        if use_memory:
            self.memory_state = new_memory
            
        # Get memory metrics
        metrics = self.predictor.model.neural_memory.get_memory_metrics(
            new_memory
        )
        
        return generated_text, metrics
        
    def create_interface(self):
        """Create Gradio interface for demo."""
        return gr.Interface(
            fn=self.generate,
            inputs=[
                gr.Textbox(lines=5, label="Input Text"),
                gr.Slider(1, 500, value=100, label="Max New Tokens"),
                gr.Slider(0.1, 2.0, value=0.8, label="Temperature"),
                gr.Checkbox(label="Use Memory State")
            ],
            outputs=[
                gr.Textbox(label="Generated Text"),
                gr.JSON(label="Memory Metrics")
            ],
            title="Titans MAC Demo",
            description="Generate text using the Titans MAC model"
        )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='base')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    
    # Initialize model and predictor
    from ..utils import ConfigLoader
    config_loader = ConfigLoader("configs")
    config = config_loader.load_model_config(args.config)
    
    # Create demo
    demo = TitansDemo(
        predictor=predictor,
        max_length=config['model']['sequence']['max_length']
    )
    
    # Launch interface
    interface = demo.create_interface()
    interface.launch(server_port=args.port)

if __name__ == '__main__':
    main()
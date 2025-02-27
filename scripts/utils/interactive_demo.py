# # titans/scripts/utils/interactive_demo.py
# import torch
# import gradio as gr
# from typing import Dict, Tuple
# from transformers import GPT2Tokenizer

# from core.models import TitansMAC
# from data.preprocessing import SequenceProcessor
# from inference.predictor import TitansPredictor
# from ...utils.config_loader import ConfigLoader
# from ...utils.logging import setup_logger
# from ...utils.config import get_config_value

# class TitansDemo:
#     def __init__(
#         self,
#         predictor: TitansPredictor,
#         max_length: int = 8192
#     ):
#         self.predictor = predictor
#         self.max_length = max_length
#         self.memory_state = None
        
#     def generate(
#         self,
#         text: str,
#         max_new_tokens: int,
#         temperature: float,
#         use_memory: bool
#     ) -> Tuple[str, Dict[str, float]]:
#         """Generate text continuation."""
#         memory = self.memory_state if use_memory else None
#         generated_text, new_memory = self.predictor.predict(
#             text,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             memory_state=memory
#         )
        
#         if use_memory:
#             self.memory_state = new_memory
            
#         # Get memory metrics
#         metrics = self.predictor.model.neural_memory.get_memory_metrics(
#             new_memory
#         )
        
#         return generated_text, metrics

# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description='Run Titans MAC Demo')
#     parser.add_argument('--checkpoint', type=str, required=True,
#                        help='Path to model checkpoint')
#     parser.add_argument('--config', type=str, default='small',
#                        help='Model configuration name')
#     parser.add_argument('--port', type=int, default=7860,
#                        help='Port for Gradio interface')
#     args = parser.parse_args()
    
#     # Setup
#     logger = setup_logger("titans_demo")
#     config_loader = ConfigLoader("configs")
#     config = config_loader.load_model_config(args.config)
    
#     # Initialize model
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = TitansMAC(
#         input_dim=get_config_value(config, 'model.input_dim', 384, int),
#         memory_dim=get_config_value(config, 'model.memory_dim', 384, int),
#         num_memory_tokens=get_config_value(config, 'model.num_memory_tokens', 16, int),
#         num_heads=get_config_value(config, 'model.num_heads', 6, int),
#         num_layers=get_config_value(config, 'model.num_layers', 2, int),
#         dropout=0.0,
#         vocab_size=get_config_value(config, 'model.vocab_size', 50000, int)
#     ).to(device)
    
#     # Load checkpoint
#     checkpoint = torch.load(args.checkpoint)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     # Setup tokenizer and processor
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     processor = SequenceProcessor(
#         tokenizer,
#         max_length=get_config_value(config, 'model.sequence.max_length', 1024, int)
#     )
    
#     # Create predictor
#     predictor = TitansPredictor(
#         model=model,
#         tokenizer=processor,
#         device=device,
#         max_length=get_config_value(config, 'model.sequence.max_length', 1024, int)
#     )
    
#     # Create demo instance
#     demo = TitansDemo(
#         predictor=predictor,
#         max_length=get_config_value(config, 'model.sequence.max_length', 1024, int)
#     )
    
#     # Create and launch interface
#     interface = gr.Interface(
#         fn=demo.generate,
#         inputs=[
#             gr.Textbox(lines=5, label="Input Text", placeholder="Enter your text here..."),
#             gr.Slider(minimum=1, maximum=500, value=100, step=1, label="Max New Tokens"),
#             gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
#             gr.Checkbox(label="Use Memory State", value=True)
#         ],
#         outputs=[
#             gr.Textbox(label="Generated Text"),
#             gr.JSON(label="Memory Metrics")
#         ],
#         title="Titans MAC Demo",
#         description="Generate text using the Titans MAC model"
#     )
    
#     # Launch the interface
#     interface.launch(server_port=args.port, share=True)

# if __name__ == '__main__':
#     main()
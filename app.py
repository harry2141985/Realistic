import gradio as gr
import spaces

import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained("glides/epicrealismxl").to("cuda")

@spaces.GPU
def generate(prompt, negative_prompt, width, height, sample_steps):
    return pipeline(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, num_inference_steps=sample_steps).images[0]

with gr.Blocks() as interface:
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", info="What do you want?", value="A perfectly red apple, 32k HDR, studio lighting", lines=4, interactive=True)
                    negative_prompt = gr.Textbox(label="Negative Prompt", info="What do you want to exclude from the image?", value="ugly, low quality", lines=4, interactive=True)
                with gr.Column():
                    generate_button = gr.Button("Generate")
                    output = gr.Image()
            with gr.Row():
                with gr.Accordion(label="Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column():
                            width = gr.Slider(label="Width", info="The width in pixels of the generated image.", value=1024, minimum=128, maximum=4096, step=64, interactive=True)
                            height = gr.Slider(label="Height", info="The height in pixels of the generated image.", value=1024, minimum=128, maximum=4096, step=64, interactive=True)
                        with gr.Column():
                            sampling_steps = gr.Slider(label="Sampling Steps", info="The number of denoising steps.", value=20, minimum=4, maximum=50, step=1, interactive=True)
        
        generate_button.click(fn=generate, inputs=[prompt, negative_prompt, width, height, sampling_steps], outputs=[output])

if __name__ == "__main__":

    interface.launch(share=True)

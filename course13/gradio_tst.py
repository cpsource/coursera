import gradio as gr

def process_text(text):
    return f"You entered: '{text}'"

# Create the Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(label="Enter some text"),
    outputs=gr.Textbox(label="Output")
)

# Launch the demo
demo.launch(share=True)


from src.chat import ChatWrapper
from src.documentsearch import update_embeddings, update_use_embeddings
from src.helper import checkbox_state_setter, reset_memory
from src.talking import transcribe
from src.constants import user_examples
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

chat = ChatWrapper()

with gr.Blocks(css=".gradio-container {background-color: #80A1D4}") as block:
    llm_state = gr.State()
    history_state = gr.State()
    chain_state = gr.State()
    verbosity_state = gr.State(False)
    speak_text_state = gr.State(False)
    memory_state = gr.State()
    embeddings_state = gr.State()
    qa_chain_state = gr.State()
    docsearch_state = gr.State()
    use_embeddings_state = gr.State(False)

    use_gpt4_state = gr.State(False)

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column():
                gr.HTML("""<b><center>GPT Assistant</center></b>""")

        with gr.Row():
            with gr.Column(scale=1):
                speak_text_checkbox = gr.Checkbox(
                    label="Enable speech", value=False)
                speak_text_checkbox.change(checkbox_state_setter, inputs=[speak_text_checkbox, speak_text_state],
                                           outputs=[speak_text_state])
                tmp_aud_file = gr.File("audios/tempfile.mp3", visible=False)
                tmp_aud_file_url = "/file=" + tmp_aud_file.value['name']
                htm_audio = f'<audio><source src={tmp_aud_file_url} type="audio/mp3"></audio>'
                audio_html = gr.HTML(htm_audio)

        with gr.Row():
            chatbot = gr.Chatbot()

        with gr.Row():
            message = gr.Textbox(label="What's on your mind?",
                                 placeholder="What's the answer to life, the universe, and everything?", lines=2)
            audio_comp = gr.Microphone(source="microphone", type="filepath", label="Just say it!",
                                       interactive=True, streaming=False)
            audio_comp.change(transcribe, inputs=[
                audio_comp], outputs=[message])
            submit = gr.Button(value="Send", variant="secondary").style(
                full_width=False)

        with gr.Accordion("General examples", open=True):
            gr.Examples(
                examples=user_examples,
                inputs=message
            )

    with gr.Tab("Settings"):
        trace_chain_checkbox = gr.Checkbox(
            label="Show reasoning chain in chat bubble", value=False)
        trace_chain_checkbox.change(checkbox_state_setter, inputs=[trace_chain_checkbox, verbosity_state],
                                    outputs=[verbosity_state])

        embeddings_text_box = gr.Textbox(label="Enter text for embeddings and hit Create:",
                                         lines=20)

        with gr.Row():
            use_embeddings_checkbox = gr.Checkbox(
                label="Use embeddings", value=False)
            use_embeddings_checkbox.change(update_use_embeddings, inputs=[use_embeddings_checkbox, use_embeddings_state],
                                           outputs=[use_embeddings_state])

            embeddings_text_submit = gr.Button(
                value="Create", variant="secondary").style(full_width=False)
            embeddings_text_submit.click(update_embeddings,
                                         inputs=[embeddings_text_box,
                                                 embeddings_state],
                                         outputs=[docsearch_state])

        reset_btn = gr.Button(value="Reset chat",
                              variant="secondary").style(full_width=False)
        reset_btn.click(reset_memory, inputs=[history_state, memory_state],
                        outputs=[chatbot, history_state, memory_state])

    message.submit(chat, inputs=[
        message, history_state, verbosity_state, speak_text_state, qa_chain_state, docsearch_state, use_embeddings_state,
    ],
        outputs=[
        chatbot, history_state, audio_html, tmp_aud_file, message
    ])

    submit.click(chat, inputs=[
        message, history_state, verbosity_state, speak_text_state, qa_chain_state, docsearch_state, use_embeddings_state,
    ],
        outputs=[chatbot, history_state, audio_html, tmp_aud_file, message])

block.launch(debug=True)

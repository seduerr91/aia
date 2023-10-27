
import os
from contextlib import closing
from typing import Optional, Tuple
import boto3
import gradio as gr

import whisper

from langchain import ConversationChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from threading import Lock
from io import StringIO
import sys
import re
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain


load_dotenv()
MAX_TOKENS = 512
TOOLS_DEFAULT_LIST = ['pal-math']
WHISPER_MODEL = whisper.load_model("tiny")
USE_GPT4_DEFAULT = False


def transcribe(aud_inp):
    if aud_inp is None:
        return ""
    aud = whisper.load_audio(aud_inp)
    aud = whisper.pad_or_trim(aud)
    mel = whisper.log_mel_spectrogram(aud).to(WHISPER_MODEL.device)

    options = whisper.DecodingOptions(fp16=False)

    result = whisper.decode(WHISPER_MODEL, mel, options)
    print("result.text", result.text)
    result_text = ""
    if result and result.text:
        result_text = result.text
    return result_text


def load_chain(tools_list, llm):
    chain = None
    memory = None
    if llm:
        tool_names = tools_list
        tools = load_tools(tool_names, llm=llm)
        memory = ConversationBufferMemory(memory_key="chat_history")
        chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,
                                 memory=memory)
    return chain, memory


def run_chain(chain, inp, capture_hidden_text):
    hidden_text = None
    output = ""
    os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
    if capture_hidden_text:
        tmp = sys.stdout
        hidden_text_io = StringIO()
        sys.stdout = hidden_text_io
        output = chain.run(input=inp)
        sys.stdout = tmp
        hidden_text = hidden_text_io.getvalue()

        # remove escape characters from hidden_text
        hidden_text = re.sub(r'\x1b[^m]*m', '', hidden_text)
        hidden_text = re.sub(
            r"Entering new AgentExecutor chain...\n", "", hidden_text)
        hidden_text = re.sub(r"Finished chain.", "", hidden_text)
        hidden_text = re.sub(r"Thought:", "\n\nThought:", hidden_text)
        hidden_text = re.sub(r"Action:", "\n\nAction:", hidden_text)
        hidden_text = re.sub(r"Observation:", "\n\nObservation:", hidden_text)
        hidden_text = re.sub(r"Input:", "\n\nInput:", hidden_text)
        hidden_text = re.sub(r"AI:", "\n\nAI:", hidden_text)
    else:
        output = chain.run(input=inp)
    return output, hidden_text


def reset_memory(history, memory):
    memory.clear()
    history = []
    return history, history, memory


def do_html_audio_speak(words_to_speak):
    polly_client = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    ).client('polly')
    voice_id = "Joanna"
    language_code = "en-US"
    engine = "neural"
    response = polly_client.synthesize_speech(
        Text=words_to_speak,
        OutputFormat='mp3',
        VoiceId=voice_id,
        LanguageCode=language_code,
        Engine=engine
    )
    html_audio = '<pre>no audio</pre>'
    if "AudioStream" in response:
        with closing(response["AudioStream"]) as stream:

            try:
                with open('audios/tempfile.mp3', 'wb') as f:
                    f.write(stream.read())
                temp_aud_file = gr.File("audios/tempfile.mp3")
                temp_aud_file_url = "/file=" + temp_aud_file.value['name']
                html_audio = f'<audio autoplay><source src={temp_aud_file_url} type="audio/mp3"></audio>'
            except IOError as error:
                print(error)
                return None, None
    else:
        print("Could not stream audio")
        return None, None

    return html_audio, "audios/tempfile.mp3"


class ChatWrapper:

    def __init__(self):
        self.lock = Lock()

    def __call__(
            self,
            inp: str,
            history: Optional[Tuple[str, str]],
            chain: Optional[ConversationChain],
            trace_chain: bool,
            speak_text: bool,
            qa_chain,
            docsearch,
            use_embeddings
    ):

        print("inp: " + inp)
        print("speak_text: ", speak_text)
        print('chain: ', chain)
        print('trace_chain: ', trace_chain)
        llm = ChatOpenAI(temperature=0, max_tokens=MAX_TOKENS,
                         model_name="gpt-3.5-turbo")
        chain, memory = load_chain(TOOLS_DEFAULT_LIST, llm)
        embeddings = OpenAIEmbeddings()

        # qa_chain = load_qa_chain(ChatOpenAI(
        # temperature=0, model_name="gpt-4"), chain_type="stuff")
        qa_chain = load_qa_chain(ChatOpenAI(
            temperature=0, model_name="gpt-3.5-turbo"), chain_type="stuff")

        history = history or []
        self.lock.acquire()
        try:
            if chain:
                # Set OpenAI key
                import openai
                openai.api_key = os.environ.get('OPENAI_API_KEY')
                if use_embeddings:
                    if inp and inp.strip() != "":
                        if docsearch:
                            docs = docsearch.similarity_search(inp)
                            output = str(qa_chain.run(
                                input_documents=docs, question=inp))
                        else:
                            output, hidden_text = "Please supply some text in the the Embeddings tab.", None
                    else:
                        output, hidden_text = "What's on your mind?", None
                else:
                    complete_inp = inp
                    output, hidden_text = run_chain(
                        chain, inp=complete_inp, capture_hidden_text=trace_chain)
            text_to_display = output
            if trace_chain:
                text_to_display = hidden_text + "\n\n" + output
            history.append((inp, text_to_display))
            if speak_text:
                html_audio, temp_aud_file = do_html_audio_speak(output)
            else:
                pass

        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history, html_audio, temp_aud_file, ""


chat = ChatWrapper()


def update_foo(widget, state):
    if widget:
        state = widget
        return state

# Pertains to question answering functionality


def update_embeddings(embeddings_text, embeddings):
    if embeddings_text:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(embeddings_text)

        docsearch = FAISS.from_texts(texts, embeddings)
        print("Embeddings updated")
        return docsearch


# Pertains to question answering functionality
def update_use_embeddings(widget, state):
    if widget:
        state = widget
        return state


with gr.Blocks(css=".gradio-container {background-color: lightgray}") as block:
    llm_state = gr.State()
    history_state = gr.State()
    chain_state = gr.State()
    express_chain_state = gr.State()
    tools_list_state = gr.State(TOOLS_DEFAULT_LIST)
    trace_chain_state = gr.State(False)
    speak_text_state = gr.State(False)
    # Takes the input and repeats it back to the user, optionally transforming it.
    monologue_state = gr.State(False)
    memory_state = gr.State()

    # Pertains to question answering functionality
    embeddings_state = gr.State()
    qa_chain_state = gr.State()
    docsearch_state = gr.State()
    use_embeddings_state = gr.State(False)

    use_gpt4_state = gr.State(USE_GPT4_DEFAULT)

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column():
                gr.HTML("""<b><center>GPT Assistant</center></b>""")

        with gr.Row():
            with gr.Column(scale=1):
                speak_text_cb = gr.Checkbox(label="Enable speech", value=False)
                speak_text_cb.change(update_foo, inputs=[speak_text_cb, speak_text_state],
                                     outputs=[speak_text_state])
                tmp_aud_file = gr.File("audios/tempfile.mp3", visible=False)
                tmp_aud_file_url = "/file=" + tmp_aud_file.value['name']
                htm_audio = f'<audio><source src={tmp_aud_file_url} type="audio/mp3"></audio>'
                audio_html = gr.HTML(htm_audio)

            with gr.Column(scale=7):
                chatbot = gr.Chatbot()

        with gr.Row():
            message = gr.Textbox(label="What's on your mind?",
                                 placeholder="What's the answer to life, the universe, and everything?", lines=2)
            submit = gr.Button(value="Send", variant="secondary").style(
                full_width=False)

        with gr.Row():
            audio_comp = gr.Microphone(source="microphone", type="filepath", label="Just say it!",
                                       interactive=True, streaming=False)
            audio_comp.change(transcribe, inputs=[
                              audio_comp], outputs=[message])

        with gr.Accordion("General examples", open=False):
            gr.Examples(
                examples=["How many people live in Canada?",
                          "What is 2 to the 30th power?",
                          "If x+y=10 and x-y=4, what are x and y?",
                          "How much did it rain in SF today?",
                          ],
                inputs=message
            )
        with gr.Accordion("Language practice examples", open=False):
            gr.Examples(
                examples=[
                    "Let's play three truths and a lie",
                    "Let's play a game of rock paper scissors",
                ],
                inputs=message
            )

    with gr.Tab("Settings"):
        trace_chain_cb = gr.Checkbox(
            label="Show reasoning chain in chat bubble", value=False)
        trace_chain_cb.change(update_foo, inputs=[trace_chain_cb, trace_chain_state],
                              outputs=[trace_chain_state])

        reset_btn = gr.Button(value="Reset chat",
                              variant="secondary").style(full_width=False)
        reset_btn.click(reset_memory, inputs=[history_state, memory_state],
                        outputs=[chatbot, history_state, memory_state])

    with gr.Tab("Embeddings"):
        embeddings_text_box = gr.Textbox(label="Enter text for embeddings and hit Create:",
                                         lines=20)

        with gr.Row():
            use_embeddings_cb = gr.Checkbox(
                label="Use embeddings", value=False)
            use_embeddings_cb.change(update_use_embeddings, inputs=[use_embeddings_cb, use_embeddings_state],
                                     outputs=[use_embeddings_state])

            embeddings_text_submit = gr.Button(
                value="Create", variant="secondary").style(full_width=False)
            embeddings_text_submit.click(update_embeddings,
                                         inputs=[embeddings_text_box,
                                                 embeddings_state, qa_chain_state],
                                         outputs=[docsearch_state])
    message.submit(chat, inputs=[
        message, history_state, chain_state, trace_chain_state, speak_text_state, qa_chain_state, docsearch_state, use_embeddings_state,
    ],
        outputs=[
        chatbot, history_state, audio_html, tmp_aud_file, message
    ])

    submit.click(chat, inputs=[
        message, history_state, chain_state, trace_chain_state, speak_text_state, qa_chain_state, docsearch_state, use_embeddings_state,
    ],
        outputs=[chatbot, history_state, audio_html, tmp_aud_file, message])

block.launch(debug=True)

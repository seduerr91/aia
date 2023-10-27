import os
from typing import Optional, Tuple
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from .chains import get_agent, run_agent
from .constants import MAX_TOKENS
from .talking import pronounciate


class ChatWrapper:

    def __call__(
            self,
            inp: str,
            history: Optional[Tuple[str, str]],
            verbosity: bool,
            speak_text: bool,
            qa_chain,
            docsearch,
            use_embeddings: bool,
    ):
        llm = ChatOpenAI(temperature=0, max_tokens=MAX_TOKENS)
        agent = get_agent(llm)
        qa_chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")

        history = history or []
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        if agent:
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
                output, hidden_text = run_agent(
                    agent, inp=inp, capture_hidden_text=verbosity)
            text_to_display = output
        if verbosity:
            text_to_display = hidden_text + "\n\n" + output
        history.append((inp, text_to_display))
        html_audio = None
        temp_aud_file = None
        if speak_text:
            html_audio, temp_aud_file = pronounciate(output)
        return history, history, html_audio, temp_aud_file, ""


import os
from contextlib import closing
import boto3
import gradio as gr
import whisper
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from rich import print

load_dotenv()


def transcribe(aud_inp: str):
    if aud_inp is None:
        return ""
    model = whisper.load_model("tiny")
    aud = whisper.load_audio(aud_inp)
    aud = whisper.pad_or_trim(aud)
    mel = whisper.log_mel_spectrogram(aud).to(model.device)

    options = whisper.DecodingOptions(fp16=False)

    result = whisper.decode(model, mel, options)
    print("result.text", result.text)
    result_text = ""
    if result and result.text:
        result_text = result.text
    return result_text


def pronounciate(words_to_speak: str):
    """Connect to Amazon Polly to turn strings into speak

    Args:
        words_to_speak (_type_): str

    Returns:
        _type_: _description_
    """
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

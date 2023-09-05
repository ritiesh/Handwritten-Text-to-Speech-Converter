import json
import cv2
import gradio as gr
import speech_recognition as sr
from gtts import gTTS
import IPython.display as ipd
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

# Load the word list and create the prefix tree
with open('C:\\Users\\91934\\Desktop\\s7 project\\text to speech\\data\\words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)

# Speech recognition setup
recognizer = sr.Recognizer()

def process_page(img, height, enlarge, use_dictionary, min_words_per_line, text_scale):
    read_lines = read_page(img,
                           detector_config=DetectorConfig(height=height, enlarge=enlarge),
                           line_clustering_config=LineClusteringConfig(min_words_per_line=min_words_per_line),
                           reader_config=ReaderConfig(decoder='word_beam_search' if use_dictionary else 'best_path',
                                                      prefix_tree=prefix_tree))

    # create text to show
    res = ''
    for read_line in read_lines:
        res += ' '.join(read_word.text for read_word in read_line) + '\n'

    # create visualization to show
    for i, read_line in enumerate(read_lines):
        for read_word in read_line:
            aabb = read_word.aabb
            cv2.rectangle(img,
                          (aabb.xmin, aabb.ymin),
                          (aabb.xmin + aabb.width, aabb.ymin + aabb.height),
                          (255, 0, 0),
                          2)
            cv2.putText(img,
                        read_word.text,
                        (aabb.xmin, aabb.ymin + aabb.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        color=(255, 0, 0))

    return res, img

def speak_text(text):
    tts = gTTS(text)
    tts.save("output.mp3")
    return ipd.Audio("output.mp3")

with open('C:\\Users\\91934\\Desktop\\s7 project\\text to speech\\data\\config.json') as f:
    config = json.load(f)

examples = []
for k, v in config.items():
    examples.append([f'C:\\Users\\91934\\Desktop\\s7 project\\text to speech\\data{k}', v['height'], v['enlarge'], False, 2, v['text_scale']])
# Define gradio interface
def gr_interface(img, height, enlarge, use_dictionary, min_words_per_line, text_scale):
    res, img = process_page(img, height, enlarge, use_dictionary, min_words_per_line, text_scale)
    speak_button = f"<button onclick=\"speakText('{res}')\">Speak Text</button>"
    return f"{res}<br>{speak_button}", img

iface = gr.Interface(fn=gr_interface,
                     inputs=[gr.Image(label='Input image'),
                             gr.Slider(10, 2000, 1000, label='Image height'),
                             gr.Slider(0, 25, 1, step=1, label='Enlarge detection'),
                             gr.Checkbox(value=False, label='Use dictionary'),
                             gr.Slider(1, 10, 1, step=1, label='Minimum number of words per line'),
                             gr.Slider(0.5, 2, 1, label='Text size in visualization')],
                     outputs=[gr.HTML(label='Read Text and Speak', type='label'), gr.Image(label='Visualization')],
                     # examples=examples,
                     allow_flagging='never',
                     title='Detect and Read Handwritten Words',
                     theme=gr.themes.Monochrome())

iface.launch()

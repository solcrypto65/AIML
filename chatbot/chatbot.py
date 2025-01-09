import gradio as gr
from openai import OpenAI, OpenAIError
import os
from IPython.display import display, Markdown, Latex, HTML, JSON

messages = [
	{'role' : 'system', 'content' : 'You are a friendly chatbot.'}
#	{'role' : 'user', 'content' : 'what is human genomics project'}
]

css = """
	.chatHistory {background-color: #99ccff !important}
	.promptBox {background-color: #66ccff !important}
"""

def gptResponse(prompt: str):
#	newMessage = {'role' : 'user', 'content' : prompt}
	messages.append({'role' : 'user', 'content' : prompt})
	resp = get_completion(messages)
	messages.append({'role' : 'assistant', 'content' : resp})
	displayMessage = ''

	for item in messages:
		if item['role'] != 'system':
			displayMessage += item['role'] + ' : ' + item['content'] + '\n\n'

	return displayMessage

def main():
	
	OpenAI.api_key  = 'my api key'

#	demo = gr.Interface(
#		fn=gptResponse,
#		inputs=['text'],
#		outputs=['text'],
#		title="!!! I am your friendly assistant !!!",
#		description="How can I help you today ?"
#	)

	with gr.Blocks(css=css) as demo:
		gr.Markdown(
			"""
			#	!!! Hi, I am your friendly assistant !!!
				        how may I help you today ?
			""")
		prompt = gr.Textbox(label='Ask your question', elem_classes='promptBox')
		output = gr.Textbox(label='Chat History', elem_classes='chatHistory')
		qryBtn = gr.Button('Ask', elem_classes='promptBox')
		qryBtn.click(fn=gptResponse, inputs=prompt, outputs=output, api_name='ask')

	demo.launch()

	messages.append({'role' : 'assistant', 'content' : resp})
	messages.append({'role' : 'user', 'content' : 'explain how this has helped in understanding human health' })

	resp = get_completion(messages)
	messages.append({'role' : 'assistant', 'content' : resp})

	print(messages)


def get_completion(messages, model="gpt-3.5-turbo", temperature=0):
    
    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"Error: {e}"

if __name__== "__main__":
   main()


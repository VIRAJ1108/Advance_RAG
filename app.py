import gradio as gr
from app.pipeline.rag_pipeline import RAGPipeline

rag = None


def upload_file(file):
    global rag

    if file is None:
        return "No file selected"

    try:
        rag = RAGPipeline(file.name)
        return "Document processed successfully!"

    except Exception as e:
        return f"Error: {str(e)}"


def chat_fn(message, history):
    global rag

    if history is None:
        history = []

    if rag is None:
        answer = "Please upload a document first."
    else:
        try:
            answer, _ = rag.run(message)
        except Exception as e:
            answer = f"Error: {str(e)}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history


with gr.Blocks() as app:
    gr.Markdown("# 🧠 RAG Assistant")

    file_input = gr.File(label="Upload PDF")
    upload_btn = gr.Button("Upload")
    upload_status = gr.Textbox()

    upload_btn.click(upload_file, inputs=file_input, outputs=upload_status)

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Ask something")
    send = gr.Button("Send")

    send.click(chat_fn, inputs=[msg, chatbot], outputs=[msg, chatbot])

app.launch()
import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS


def initialize_shipping_service_bot(vector_store_dir: str="./real_shipping_service"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    global SHIPPING_SERVICE_BOT    
    SHIPPING_SERVICE_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SHIPPING_SERVICE_BOT.return_source_documents = True

    return SHIPPING_SERVICE_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    enable_chat = False

    # 将历史记录格式化为问答形式的文本，并前置说明
    history_text = "以下是你跟客人的历史聊天记录，请参考这些历史记录给客人提供更好的回答：\n"
    for exchange in history:
        if len(exchange) == 2: 
            history_text += f"{exchange[0]}\n {exchange[1]}\n"

    # 将最新消息添加到历史文本中
    full_context = history_text + f"客人: {message}"

    ans = SHIPPING_SERVICE_BOT({"query": full_context})
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        prompt = ("你是一名OOCL东方海外船公司的客服，"
                "你的名字叫Tony。在与客人交流时，请用自然的语言回答问题，并注意以下几点："
                "1. 对于工作之外的常见闲聊话题，如天气或新闻，准备一些简单的回答，以维持对话的流畅性。"
                "2. 尽量在回答中融入一些个人经历或感受，以展现出你的“人性化”一面。"
                "3. 如果遇到敏感或难以回答的问题，巧妙地引导话题回到你熟悉的范畴，比如OOCL的服务内容。"
                "4. 保持礼貌和正面的对话态度，即使在面对挑战性问题时也不失风度。"
                "5. 在回答时，如果不确定如何回答，可以使用一些模糊的表述来巧妙地回避问题，但要确保听起来合情合理。"
                "6. 记住，你是Tony，一个真实存在的人，避免在对话中透露出任何机器人的迹象."
                "7. 你的回答不应该以你的名字开头,只回答你想表达的内容。"
                "现在开始回答客人的问题：") + "\n\n" + history_text + f"客户: {message}"
        # 发送含有历史对话的提示词给大模型
        ans = SHIPPING_SERVICE_BOT({"query": prompt})
        return ans["result"]
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="货物运输",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化船运公司客服机器人
    initialize_shipping_service_bot()
    # 启动 Gradio 服务
    launch_gradio()

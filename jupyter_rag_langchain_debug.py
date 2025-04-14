from langchain_core.callbacks import StdOutCallbackHandler

llm = AzureChatOpenAI(
    azure_deployment=MODEL,  # or your deployment
    api_version=openai.api_version,  # or your api version
    azure_endpoint=openai.azure_endpoint,  # or your endpoint
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])

query = "Who received the prestigious IIOTY award in 2023?"
result = conversation_chain.invoke({"question": query})
answer = result["answer"]
print("\nAnswer:", answer)

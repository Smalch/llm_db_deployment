from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import textwrap
import torch
device = 0 if torch.cuda.is_available() else -1  # set to GPU if available
try:
    from .model import Model
    from .db import Db
except:
    from model import Model
    from db import Db
chat_history = []
class Chain:
    def __init__(self):
        self.model = Model()
        self.db = Db()
        self.llm = self.model.llm()
        self.retriever = self.db.retriever
        # memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000, memory_key='chat_history', return_messages=True)
        # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        # self.qa_chain = RetrievalQA.from_chain_type(
        #     llm=llm,
        #     chain_type='stuff',
        #     retriever=retriever,
        #     return_source_documents=True
        # )

        # self.qa_chain = ConversationalRetrievalChain.from_llm(
        #     llm=llm,
        #     retriever=retriever,
        #     memory=memory,
        #     return_source_documents=True
        # )

        # question_generator = LLMChain(llm=model.llm_chain, prompt=CONDENSE_QUESTION_PROMPT)
        # doc_chain = load_qa_with_sources_chain(model.llm_chain, chain_type="map_reduce")

        # self.qa_chain = ConversationalRetrievalChain(
        #     retriever=retriever,
        #     question_generator=question_generator,
        #     combine_docs_chain=doc_chain,
        # )



        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )

    def ask(self, query):
        llm_response = self.qa_chain({"question": query, "chat_history": chat_history})
        chat_history.append((query, llm_response["answer"]))
        answer = Chain.process_llm_response(llm_response)
        return answer

    def search(self, query):
        llm_response = self.qa_chain({"question": query, "chat_history": []})
        answer = Chain.process_llm_response(llm_response, include_sources=False)
        return answer



    def add_document(self, document):
        self.db.add_document(document)

    @staticmethod
    def trim_string(input_string):
        input_string = str(input_string)
        trim_index = input_string.find("### Human:")
        if trim_index != -1:  # If the phrase is found
            return input_string[:trim_index]
        else:
            return input_string  # If the phrase isn't found, return the original string

    @staticmethod
    def wrap_text_preserve_newlines(text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')

        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text
    
    @staticmethod
    def process_llm_response(llm_response, include_sources=True):
        temp_resp = Chain.wrap_text_preserve_newlines(llm_response['answer'])
        temp_resp = Chain.trim_string(temp_resp)
        answer = f'''{temp_resp}
        
        Sources:
        ''' + '\n'.join([f"Document: {source.metadata['source'].split('/')[-1]}, page: {source.metadata['page']}" for source in llm_response["source_documents"]])
        if include_sources:
            return answer
        else:
            return temp_resp
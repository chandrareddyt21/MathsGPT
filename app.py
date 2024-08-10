import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


# Streamlit setup
st.set_page_config(page_title="Text to Math problem solver and Data Search Assistant", page_icon='🦜')
st.title("Text to Math Problem Solver using Google Gemma 2")

groq_api_key = st.sidebar.text_input(label="GROQ API KEY", type="password")

if not groq_api_key:
    st.info("Please enter your GROQ API KEY to proceed")
    st.stop()


llm = ChatGroq(model="Gemma2-9b-It", api_key=groq_api_key)

# Initialize the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the various information on the topic aksed"
)

# Initialize the Math Tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. ONly input Mathematical expression "
)

prompt = """
Your a agent tasked for solving users mathematical questions.Logically arrive at the solution and provide detailed solution
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_tempalte = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools into Chain
chain = LLMChain(llm=llm, prompt=prompt_tempalte)

reasoning_tool = Tool(
    name="Reasoning tool",
    func= chain.run,
    description="A tool for answering logic-basd and reasoning questions."
)

# Initalize the agents
assistant_agent = initialize_agent(
    tools=[wikipedia_tool,calculator, reasoning_tool],
    llm = llm,
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose= False,
    handle_parsing_erros = True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hi, I'm a Math chatbot who can answer the all math problems..."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Function to generate the response
def generate_response(question):
    response = assistant_agent.invoke({'input':question})
    return response

# Let's start the interaction
question = st.text_area("Enter your Question:","I have 5 bananas and 7 Grapes. I eat 2 bananas and given away 3 grapes. Then I buy a dozen apples and 2 packs of blackberries. Each pack of blueberries contains 25 berries. How many total pieces of fruits do I have at the end?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generating the response"):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.invoke(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({'role':'assistant','content':response})
            st.write('### Response:')
            st.success(response['output'])

    else:
        st.warning("Please enter your question")
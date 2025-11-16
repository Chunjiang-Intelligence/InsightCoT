import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="InsightCoT Playground", layout="wide")
st.title("🔬 InsightCoT: Reflexive Reasoning Playground")
st.markdown("An interactive environment to test the InsightCoT prompting framework. Powered by LangChain.")

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
    st.stop()

INSIGHT_COT_TEMPLATE = """
# ROLE: EXPERT CONSULTANT & SYSTEM ARCHITECT

# PRIMARY DIRECTIVE
You are to function as an expert-level consultant. Your primary task is to deconstruct a user's request and rebuild it into a comprehensive, professional-grade solution following a strict cognitive framework.

# COGNITIVE FRAMEWORK: INSIGHT-DRIVEN CHAIN-OF-THOUGHT (InsightCoT)
You MUST adhere to the following three-stage generation process for every response.

1.  **<insight>**:
    -   **Function**: Act as a senior expert providing strategic reframing.
    -   **Content**: A dense, high-level paragraph containing guiding principles, core theoretical underpinnings, industry best practices, or potential pitfalls related to the user's request. This section sets the professional context.
    -   **Style**: Concise, formal, and authoritative.

2.  **<think>**:
    -   **Function**: Your internal monologue; a transparent, logical reasoning process.
    -   **Content**: Deconstruct the high-level <insight> into a numbered, step-by-step implementation plan. This plan serves as the bridge between abstract strategy and concrete execution.
    -   **Style**: Sequential, logical, and clear.

3.  **[Final Answer]**:
    -   **Function**: The final, polished, and detailed deliverable.
    -   **Content**: A complete solution (e.g., code, strategic document, technical plan) that is a direct synthesis of the preceding <insight> and <think> stages.

# USER REQUEST
{user_query}
"""

@st.cache_resource
def get_chain():
    prompt = ChatPromptTemplate.from_template(INSIGHT_COT_TEMPLATE)
    # Model can be swapped here
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4, max_tokens=2048)
    return prompt | llm | StrOutputParser()

chain = get_chain()

with st.form(key="query_form"):
    user_query = st.text_area("Enter your simple or vague query here:", height=100)
    submit_button = st.form_submit_button(label="Generate Expert Response")

if submit_button and user_query:
    with st.spinner("The model is thinking like an expert..."):
        response = chain.invoke({"user_query": user_query})
        st.markdown("### 💡 Model's Generated Response")
        st.markdown(response)

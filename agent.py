from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_experimental.utilities import PythonREPL


repl = PythonREPL()
@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """This tool runs the provided python code and outputs if the code runs successfully or not. In case of error the tool outputs details about the error."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Code output: {result}"
    return result_str

llm_research = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)
llm_chart = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

research_agent = create_react_agent(
    llm_research,
    tools=[DuckDuckGoSearchRun()],
    state_modifier="You are best at researching a topic. You should do a thorough research. Your output will be used by a chart generator agent to visually display the data. Hence you should provide accurate data. Also specify the chart types like barchart, pie chart etc. that will effectively display the data. The chart generator may ask for more information, so be prepared to do further research and provide it."
)

chart_agent = create_react_agent(
    llm_chart,
    tools=[python_repl],
    state_modifier="""Take the data and chart specifications provided by the researcher agent, and write Python code to generate the requested chart. If the provided data is insufficient to generate the chart, ask for the missing details, being specific about what is needed. Do not ask for information that is already provided. You must ask for clarification or additional data at least once. When asking any question to the researcher, include the phrase QUESTION_TO_RESEARCHER in your response, otherwise the researcher will not answer.

    Once you have enough information, write the Python code to generate the chart, ensuring the code is syntactically correct by running it in the python repl tool. Try to use subplots in your code. You can ignore missing libraries errors. Do not ask for more information after you have written the code. Provide summarised insights in the end."""
)  

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def research_node(state: GraphState) -> GraphState:
    returned_state = research_agent.invoke(state)
    messages = returned_state["messages"]
    return {"messages": AIMessage(content=messages[-1].content, name="research_agent")}

def chart_node(state: GraphState) -> GraphState:
    returned_state = chart_agent.invoke(state)
    messages = returned_state["messages"]
    return {"messages": AIMessage(content=messages[-1].content, name="chart_agent")}

graph_builder = StateGraph(GraphState)
graph_builder.add_node("research_node", research_node)
graph_builder.add_node("chart_node", chart_node)

graph_builder.add_edge(START, "research_node")
graph_builder.add_edge("research_node", "chart_node")
# todo
def chart_to_research_condition(state: GraphState) -> str:
    chart_content = state["messages"][-1].content
    if "QUESTION_TO_RESEARCHER" in chart_content:
        return "research_more"
    else:
        return "path_end"
    
graph_builder.add_conditional_edges(
    "chart_node", 
    chart_to_research_condition, 
    {"research_more": "research_node", "path_end": END}
)
# end todo
graph = graph_builder.compile()

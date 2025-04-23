# bee_data_agent.py
import bee
from bee import Agent, Tool
import pandas as pd
import joblib

# Web framework
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# BeeAI framework enhancements
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow, AgentWorkflowInput

# Base Tools
class LoadData(Tool):
    name = "load_data"
    description = "Load data from uploaded CSV into a DataFrame."
    def run(self, file_bytes: bytes) -> pd.DataFrame:
        from io import BytesIO
        return pd.read_csv(BytesIO(file_bytes))

class SummarizeData(Tool):
    name = "summarize_data"
    description = "Return summary statistics of a DataFrame."
    def run(self, df: pd.DataFrame) -> dict:
        return df.describe(include='all').to_dict()

class FilterData(Tool):
    name = "filter_data"
    description = "Filter DataFrame by column conditions."
    def run(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        return df.query(condition)

class AggregateData(Tool):
    name = "aggregate_data"
    description = "Aggregate DataFrame by group and operations."
    def run(self, df: pd.DataFrame, group_cols: list, agg_dict: dict) -> pd.DataFrame:
        return df.groupby(group_cols).agg(agg_dict)

class MLModelTool(Tool):
    name = "run_ml_model"
    description = "Apply a pre-trained ML model to feature DataFrame."
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
    def run(self, df: pd.DataFrame, feature_cols: list) -> list:
        X = df[feature_cols]
        return self.model.predict(X).tolist()

class WikipediaSearchTool(Tool):
    name = "wikipedia_search"
    description = "Fetch summary from Wikipedia for a query."
    def __init__(self):
        self.tool = WikipediaTool()
    def run(self, query: str) -> str:
        return self.tool.search(query)

class WeatherTool(Tool):
    name = "get_weather"
    description = "Fetch weather data for a location."
    def __init__(self):
        self.tool = OpenMeteoTool()
    def run(self, location: str) -> dict:
        return self.tool.get_weather(location)

# Agent
class DataAnalyticAgent(Agent):
    def __init__(self, model_path: str = None, chat_model_name: str = "gpt-4"):
        self.chat = ChatModel(model_name=chat_model_name)
        tools = [LoadData(), SummarizeData(), FilterData(), AggregateData(), WikipediaSearchTool(), WeatherTool()]
        if model_path:
            tools.append(MLModelTool(model_path))
        super().__init__(tools=tools)
        self.current_df = None

    def run(self, action: str, payload: dict):
        if action == "load":
            self.current_df = self.invoke_tool("load_data", payload['file_bytes'])
            return f"Data loaded: {len(self.current_df)} rows"
        if action == "summarize": return self.invoke_tool("summarize_data", self.current_df)
        if action == "filter":
            self.current_df = self.invoke_tool("filter_data", self.current_df, payload['condition'])
            return self.current_df.to_dict(orient='records')
        if action == "aggregate":
            result = self.invoke_tool("aggregate_data", self.current_df, payload['group_cols'], payload['agg_dict'])
            return result.to_dict(orient='records')
        if action == "predict": return self.invoke_tool("run_ml_model", self.current_df, payload['feature_cols'])
        if action == "wiki": return self.invoke_tool("wikipedia_search", payload['query'])
        if action == "weather": return self.invoke_tool("get_weather", payload['location'])
        if action == "chat": return self.chat.chat(payload['message'])
        return "Unsupported action"

# Workflow Example
def run_workflow(file_bytes: bytes, workflow_input: AgentWorkflowInput) -> any:
    agent = DataAnalyticAgent(model_path='model.joblib')
    workflow = AgentWorkflow(agent)
    return workflow.run(workflow_input)

# FastAPI WebUI
app = FastAPI()
agent = DataAnalyticAgent(model_path='model.joblib')
class QueryModel(BaseModel): action: str; params: dict
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """<html><body><h2>Data Analytic Agent</h2><form action='/upload' enctype='multipart/form-data' method='post'><input name='file' type='file' accept='.csv'><button>Upload CSV</button></form><hr/><h3>Run Query</h3><form action='/query' method='post'>Action:<input name='action'/><br/>Params(JSON):<input name='params' size='50'/><br/><button>Run</button></form></body></html>"""
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read(); msg = agent.run("load", {"file_bytes": content}); return {"message": msg}
@app.post("/query")
async def run_query(query: QueryModel): result = agent.run(query.action, query.params); return {"result": result}

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))



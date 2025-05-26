from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import io
import uuid

# Initialize the Gemini model (keep this outside the endpoint)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyBBifyF9OMufQIk3YNNMWuPVNQJXShYwF4",
    temperature=0.9,
)

# FastAPI app
app = FastAPI()

# In-memory session store for demonstration (not for production)
session_store = {}

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        session_id = str(uuid.uuid4())
        session_store[session_id] = df
        return {"session_id": session_id}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/ask_csv")
async def ask_csv(session_id: str = Form(...), question: str = Form(...)):
    try:
        if session_id not in session_store:
            raise HTTPException(status_code=404, detail="Session not found")
        df = session_store[session_id]
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True
        )
        response = agent.run(question)
        # Try to evaluate the response as a DataFrame
        try:
            # If the agent returns a DataFrame as a string, convert it
            result_df = eval(response) if "DataFrame" in response else df
        except Exception:
            result_df = df  # fallback to original

        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=result.csv"}
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
from fastapi import FastAPI, HTTPException, UploadFile
app = FastAPI()


@app.get("/health")
async def health():
    """
    A function that returns the health status of the application.

    Returns:
        dict: A dictionary containing the status of the application. The status is a string indicating that the application is running at a specific port.

    """
    return {"status": "running"}


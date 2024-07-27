import io
from fastapi import FastAPI, HTTPException, UploadFile
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from schemas import data
from utils.data_generator import DataGenerator
from models.regressor import Model
from schemas.model import PerformanceMetrics

app = FastAPI()


PORT: str = "dummy link"
@app.get("/health")
async def health():
    """
    A function that returns the health status of the application.

    Returns:
        dict: A dictionary containing the status of the application. The status is a string indicating that the application is running at a specific port.

    """
    return {"status":f"running at {PORT}"}

@app.get("/dataset")
async def generate_data(schema:data.DataGenerationSchema):
    """
    Generate random data based on the given schema.

    Args:
        schema (data.DataGenerationSchema): The schema used to generate the random data.

    Returns:
        LabelledDataSchema: The generated random data.

    TODO:
        Support dynamic schema structure
    """
    # TODO: Support dynamic schema structure
    return DataGenerator(schema).generate_random_data()

async def _readable_byte_data(file: UploadFile) -> io.StringIO:
    """
    Reads the contents of an uploaded file and returns a StringIO object containing the decoded contents.

    Args:
        file (UploadFile): The uploaded file to read.

    Returns:
        io.StringIO: A StringIO object containing the decoded contents of the file.
    """
    contents: bytes = await file.read()
    readable_contents: io.StringIO = io.StringIO(contents.decode("utf-8"))
    return readable_contents

@app.post("/train/{data}")
async def train(data: UploadFile, target: str = "income", n_estimators: int = 100, max_depth: int|None = None, random_state: int = 42):
    """
    Train a machine learning model using the provided data.

    Args:
        data (UploadFile): The training data in CSV format.
        target (str, optional): The target variable. Defaults to "income".
        n_estimators (int, optional): The number of estimators. Defaults to 100.
        max_depth (int, optional): The maximum depth of the tree. Defaults to None.
        random_state (int, optional): The random state. Defaults to 42.

    Raises:
        HTTPException: If the provided data is not in CSV format.

    Returns:
        dict: A dictionary containing the message indicating the success of the training.

    """
    if data.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files accepted.")

    file_contents = await _readable_byte_data(data)
    train_data = pd.read_csv(file_contents)

    message: str = Model(target, n_estimators, max_depth, random_state).train(train_data)
    return {"message": f"Model trained successfully. {message}"}

@app.post("/test/{data}/{model_id}")
async def test(data: UploadFile, model_id: str,  target: str = "income", n_estimators: int = 100, max_depth: int|None = None, random_state: int = 42):
    """
    Test a trained machine learning model on the provided test data.

    Args:
        data (UploadFile): The test data in CSV format.
        model_id (str): The ID of the trained model.
        target (str, optional): The target variable. Defaults to "income".
        n_estimators (int, optional): The number of estimators. Defaults to 100.
        max_depth (int|None, optional): The maximum depth of the tree. Defaults to None.
        random_state (int, optional): The random state. Defaults to 42.

    Raises:
        HTTPException: If the provided data is not in CSV format.

    Returns:
        dict: A dictionary containing the message indicating the performance of the model on the test set.
    """
   
    if data.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files accepted.")

    file_contents = await _readable_byte_data(data)
    test_data = pd.read_csv(file_contents)

    model: RandomForestRegressor

    metrics: PerformanceMetrics = Model(target, n_estimators, max_depth, random_state).test(model, test_data)
    return {"message": f"Model's performance on test set:\n {metrics}"}


@app.post("/predict/{data}/{model_id}")
async def predict(data: UploadFile, model_id: str, target: str = "income", n_estimators: int = 100, max_depth: int|None = None, random_state: int = 42):
    """
    Predict the target variable using a trained model on the provided test data.

    Parameters:
        data (UploadFile): The test data in CSV format.
        model_id (str): The ID of the trained model.
        target (str, optional): The target variable. Defaults to "income".
        n_estimators (int, optional): The number of estimators. Defaults to 100.
        max_depth (int|None, optional): The maximum depth of the tree. Defaults to None.
        random_state (int, optional): The random state. Defaults to 42.

    Raises:
        HTTPException: If the provided data is not in CSV format.

    Returns:
        dict: A dictionary containing the message indicating the performance of the model on the test set.
    """
   
    if data.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files accepted.")

    file_contents = await _readable_byte_data(data)
    test_data = pd.read_csv(file_contents)
    model: RandomForestRegressor

    prediction: pd.DataFrame = Model(target, n_estimators, max_depth, random_state).predict(model, test_data)
    return {"message": f"Model's performance on test set:\n {prediction}"}


@app.post("/deploy/{model_id}")
async def deploy(model_id: str):
   
    # find if model is good in 

    file_contents = await _readable_byte_data(data)
    test_data = pd.read_csv(file_contents)
    model: RandomForestRegressor

    prediction: pd.DataFrame = Model(target, n_estimators, max_depth, random_state).predict(model, test_data)
    return {"message": f"Model's performance on test set:\n {prediction}"}
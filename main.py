# main.py
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from KNN import KNN


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
model = KNN(K=1)

data = {"data":pd.read_csv("iris-dataset.csv")}
data_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


@app.get('/iris', response_class=HTMLResponse)
def iris(request: Request):
    global model
    gdf = data["data"].groupby('species')
    data["train"] = gdf.apply(lambda x: x.sample(n=10)).drop("species", axis=1).reset_index()

    # 3. 1で選んだ以外のデータからランダムに10個持ってきて、プレイヤーにその特徴量だけ見せて種類を当てさせる。また、2で作成したモデルで3で選んだデータに対して予測を行う。
    data["test"] = data["data"][~data["data"].index.isin(data["train"]["level_1"])].sample(n=10)
    model.fit(data["train"].drop(["level_1", "species"], axis=1), data["train"][['species']])

    return templates.TemplateResponse(
        "iris.html",
        {
            "request": request,
            "train": data["train"][data_columns].values,
            "test": data["test"].drop("species", axis=1).values,
        }
    )

@app.post('/iris', response_class=HTMLResponse)
def iris(request: Request, pred: list = Form(None)):
    try:
        result = pd.DataFrame([pred, model.predict(data["test"].drop("species", axis=1)), data["test"]["species"]], index=["pred_you", "pred_ai", "true"]).T
        result["あなたの正誤"] = result["pred_you"] == result["true"]
        result["AIの正誤"] = result["pred_ai"] == result["true"]

        accuracy_you = sum(result['あなたの正誤'])/10
        accuracy_ai = sum(result['AIの正誤'])/10
        winner = 'あなた' if accuracy_you > accuracy_ai else '引き分け' if accuracy_you == accuracy_ai else 'AI'

        result = result.replace(True, "⭕️").replace(False, "✖️")
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "result": result.values,
                "accuracy_you": accuracy_you,
                "accuracy_ai": accuracy_ai,
                "winner": winner,
            }
        )
    
    except:
        return templates.TemplateResponse(
            "iris.html",
            {
                "request": request,
                "train": data["train"][data_columns].values,
                "test": data["test"].drop("species", axis=1).values,
                "error_flag": True
            }
        )
    
# @app.post('/iris', response_class=HTMLResponse)
# def iris(request: Request, sepal_length: float = Form(None), sepal_width: float = Form(None), petal_length: float = Form(None), petal_width: float = Form(None)):
#     try:
#         predict = model.predict([[sepal_length,sepal_width, petal_length, petal_width]])
#         return templates.TemplateResponse(
#             "iris.html",
#             {
#                 "request": request,
#                 "predict": predict
#             }
#         )
#     except:
#         return templates.TemplateResponse(
#             "iris.html",
#             {
#                 "request": request,
#                 "error_flag": True
#             }
#         )


# http://127.0.0.1:8000/?x1=2&x2=2&x3=2
@app.get("/index/{id}", response_class=HTMLResponse)
def get_product(id: str, request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "id": id
        }
    )


@app.post('/search')
def search(q: str):
    return {'query': q}


if __name__ == '__main__':
    uvicorn.run(app)
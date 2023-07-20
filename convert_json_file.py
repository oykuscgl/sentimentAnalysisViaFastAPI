from fastapi_endpoints import FastAPI, File, UploadFile
import csv
import codecs

app = FastAPI()

@app.post("/upload")
def upload(file: UploadFile = File('cleaned_data.csv')):
    csvReader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
    data = {}
    for rows in csvReader:
        key = rows['Id']  # Assuming a column named 'Id' to be the primary key
        data[key] = rows

    file.file.close()
    return data
uvicorn --host 0.0.0.0 --port 14001 --reload rf:app &
uvicorn --host 0.0.0.0 --port 14002 --reload resnet:app &

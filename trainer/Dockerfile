FROM python:3.9

ADD requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

ADD bank_marketing__xgboost_v0__trainer.py .
CMD ["python", "./bank_marketing__xgboost_v0__trainer.py"]
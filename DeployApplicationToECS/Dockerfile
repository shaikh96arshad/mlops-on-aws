FROM python:slim

# Create app directory
WORKDIR /app

# Install app dependencies
COPY requirements.txt ./

RUN pip install -r requirements.txt

# Bundle app source
COPY DeployApplicationToECS/ .

CMD ["python", "train.py" ]

EXPOSE 80
CMD [ "python", "app.py"]
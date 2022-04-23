**Project Architecture**
![Project Architecture](https://github.com/shaikh96arshad/AWS-Projects/blob/main/Dry%20Beans/images/Architecture.png)

**Context**:
This project aims to deploy real time Mlops infrastructure on AWS cloud using decoupled services.
In this project we predict a dry beans class based on differnt features provided in input.

**The components of the reference architecture diagram are:**
1.**AWS VPC** A secured development environment was implemented using an Amazon SageMaker Notebook Instance deployed to a custom virtual private cloud (VPC), and secured by implementing security groups and routing the notebook’s internet traffic via the custom VPC.

2.**AWS S3**: Used to store datasets, training job results, training checkpoint , EDA results and model artifacts generated after training job.

3.**AWS Sagemaker** : Used AWS Sagemaker notebooks to do Exploratory Data Analysis, clean, preprocess and prepare data for model training.

4.**API Gateway** : Used as an entrypoint to serve request. API gateway captures user payload and passes it to Lamda for processing

5.**AWS Lambda** : Invokes the Sagemaker endpoint based on the payload received by API gateway.

6.**Sagemaker Endpoint** : The trained model is deployed and available to serve the request via an endpoint. AWS does all the heavy lifting of hosting the model on servers and making it highly available. Container Image and Endpoint Configurations are required to be setup Sagemaker endpoint.


"""Example workflow pipeline script for abalone pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.sklearn.processing import SKLearnProcessor

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
    


BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )



def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags




def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="mob",
    pipeline_name="mob",
    base_job_prefix="mob",
):
    
    """Gets a SageMaker ML Pipeline instance working with on Bank Personal Loan prediction data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
        
    processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount",
    default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputData",
        default_value=input_data_uri,
    )
    batch_data = ParameterString(
        name="BatchData",
        default_value=batch_data_uri,
    )
    
    
    framework_version = "0.23-1"
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name="sklearn-mob-process",
        role=role,
    )
    
    step_process = ProcessingStep(
        name="PreProcessMob",
        processor=sklearn_processor,
        inputs=[
          ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),  
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data],
    )

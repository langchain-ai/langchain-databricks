"""
This file contains the integration test for DatabricksVectorSearch class.

We run the integration tests nightly by the trusted CI/CD system defined in
a private repository, in order to securely run the tests. With this design,
integration test is not intended to be run manually by OSS contributors.
If you want to update the DatabricksVectorSearch implementation and you think
that you need to update the corresponding integration test, please contact to
the maintainers of the repository to verify the changes.
"""

import os
import time

import pytest
import requests


@pytest.mark.timeout(3600)
def test_vectorstore():
    """
    We run the integration tests for vector store by Databricks Workflow,
    because the setup is too complex to run within a single python file.
    Thereby, this test simply triggers the workflow by calling the REST API.
    """
    required_env_vars = ["DATABRICKS_HOST", "DATABRICKS_TOKEN", "VS_TEST_JOB_ID"]
    for var in required_env_vars:
        assert os.getenv(var), f"Please set the environment variable {var}."

    test_endpoint = os.getenv("DATABRICKS_HOST")
    test_job_id = os.getenv("VS_TEST_JOB_ID")
    headers = {
        "Authorization": f"Bearer {os.getenv('DATABRICKS_TOKEN')}",
    }

    # Check if there is any ongoing job run
    response = requests.get(
        f"{test_endpoint}/api/2.1/jobs/runs/list",
        json={
            "job_id": test_job_id,
            "active_only": True,
        },
        headers=headers,
    )
    no_active_run = len(response.json().get("runs", [])) == 0
    assert no_active_run, "There is an ongoing job run. Please wait for it to complete."

    # Trigger the workflow
    # TODO: We are going to replace this with the Databricks SDK once the vector store
    # class is also migrated to the SDK.
    response = requests.post(
        f"{test_endpoint}/api/2.1/jobs/run-now",
        json={
            "job_id": test_job_id,
        },
        headers=headers,
    )

    assert response.status_code == 200, "Failed to trigger the workflow."

    job_url = f"{test_endpoint}/jobs/{test_job_id}/runs/{response.json()['run_id']}"
    print(f"Started the job at {job_url}")  # noqa: T201

    # Wait for the job to complete
    while True:
        response = requests.get(
            f"{test_endpoint}/api/2.1/jobs/runs/get",
            json={
                "run_id": response.json()["run_id"],
            },
            headers=headers,
        )

        assert response.status_code == 200, "Failed to get the job status."

        status = response.json()["status"]
        if status["state"] == "TERMINATED":
            if status["termination_details"]["type"] == "SUCCESS":
                break
            else:
                assert False, "Job failed. Please check the logs in the workspace."

        time.sleep(60)
        print("Job is still running...")  # noqa: T201

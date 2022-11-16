#!/bin/bash

set -o errexit
set -o nounset

mlflow server \
--backend-store-uri="sqlite:////project/mlflow/mlflow.db" \
--default-artifact-root="/project/mlflow/artifacts" \
--host 0.0.0.0 \
--port 6000 \
--workers 2

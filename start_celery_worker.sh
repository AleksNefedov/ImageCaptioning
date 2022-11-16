#!/bin/bash

set -o errexit
set -o nounset

celery -A server.celery_app worker --loglevel=INFO --pool=solo --concurrency=4

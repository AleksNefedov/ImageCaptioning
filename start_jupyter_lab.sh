#!/bin/bash

set -o errexit
set -o nounset

jupyter lab \
--ip 0.0.0.0 \
--port 4000 \
--no-browser \
--ServerApp.token='' \
--ServerApp.password='' \
--allow-root

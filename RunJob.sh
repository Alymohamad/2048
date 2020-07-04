#!/bin/sh

LATEST_DEPOLY_DIR=$(ls -td /opt/codedeploy-agent/deployment-root/b1281a69-c307-4449-9d7d-62a3f0935d31/*/deployment-archive/ | head -1)
python3 "$LATEST_DEPOLY_DIR"/main_keras_ddqn_test.py


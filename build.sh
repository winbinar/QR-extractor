#!/usr/bin/env bash
# exit on error
set -o errexit

apt-get update
apt-get install -y libzbar0

pip install -r requirements.txt

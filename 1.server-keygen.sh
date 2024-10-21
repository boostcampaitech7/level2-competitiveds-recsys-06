#!/bin/bash
source .env

ssh server$server_no << ENDSSH
#mkdir "$data_log_path"
mkdir -p /data/ephemeral/home/data/output
git config --global user.name "$username"
git config --global user.email "$email"
git config --global --list
ENDSSH

echo "모든 작업이 완료되었습니다."
read



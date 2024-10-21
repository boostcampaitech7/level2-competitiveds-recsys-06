
scp -r config.yaml server4:/data/ephemeral/home/level2-competitiveds-recsys-06

ssh server4 << 'ENDSSH'
cd /data/ephemeral/home/level2-competitiveds-recsys-06
docker-compose restart
ENDSSH

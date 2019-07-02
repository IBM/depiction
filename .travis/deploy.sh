 #!/bin/sh

echo ${DOCKER_PASSWORD} | docker login -u ${DOCKER_USERNAME} --password-stdin
docker tag drugilsberg/depiction:latest drugilsberg/depiction:${TRAVIS_COMMIT}
docker push drugilsberg/depiction:${TRAVIS_COMMIT}
docker push drugilsberg/depiction:latest
.PHONY: train run build docker-run push kind-apply k8s-apply hpa

train:
	python scripts/train_model.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

build:
	docker build -t medical-cv-serve:local --build-arg BUILD_REV=$$(git rev-parse --short HEAD) .

docker-run:
	docker run --rm -p 8000:8000 -e BUILD_REV=$$(git rev-parse --short HEAD) medical-cv-serve:local

k8s-apply:
	kubectl apply -f k8s/k8s-namespace.yaml && \	kubectl apply -f k8s/k8s-deployment.yaml && \	kubectl apply -f k8s/k8s-service.yaml && \	kubectl apply -f k8s/k8s-hpa.yaml

hpa:
	kubectl get hpa -n ml

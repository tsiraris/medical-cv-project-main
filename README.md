# medical-cv-serve (baseline)

## Quickstart

```bash
# 1) train a tiny placeholder model
make train

# 2) run locally
make run
curl -s localhost:8000/healthz | jq
curl -s -X POST localhost:8000/predict   -H 'Content-Type: application/json'   -d '{"items":[{"f1":5.1,"f2":3.5,"f3":1.4,"f4":0.2}]}' | jq

# 3) docker build & run
make build && make docker-run

# 4) push via GitHub Actions -> GHCR (on push to main)
# check the workflow and image in ghcr.io

# 5) Kubernetes
kubectl apply -f k8s/k8s-namespace.yaml
kubectl create secret docker-registry ghcr-secret   --docker-server=ghcr.io   --docker-username=YOUR_GH_USERNAME   --docker-password=YOUR_GITHUB_TOKEN   -n ml
make k8s-apply
kubectl -n ml port-forward svc/medical-cv-serve 8080:80
curl -s localhost:8080/healthz | jq
```

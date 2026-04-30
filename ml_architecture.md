# ML Architecture Reference

Visual notes from learning how to deploy a machine learning project end-to-end — from raw data in a database to a live API on the cloud.

---

## 1. How does everything connect? Where does my DB, code, model, Docker, and AWS fit?

![ML Project Architecture](figures/ml_project_architecture.jpg)

---

## 2. What does GitHub store vs Docker Hub vs AWS? Why do I need all three?

![What Each Service Stores](figures/what_each_service_stores.jpg)

---

## 3. How does traffic actually flow through my app — rate limiting, workers, load balancer, Kubernetes?

![Traffic Management Full Picture](figures/traffic_management_full_picture.jpg)

---

## 4. What are the 5 traffic layers and why do I care about each one?

![Traffic Layers with Reasons](figures/traffic_layers_with_reasons.jpg)

---

## 5. When something goes wrong (slow, crashing, queuing), which layer fixes it?

![Bottlenecks and Solutions](figures/bottlenecks_and_solutions.jpg)

---

## 6. How do Kubernetes and Terraform fit into the ML deployment picture?

![Kubernetes and Terraform in the ML Stack](figures/kubernetes_terraform_in_ml_stack.jpg)

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  default = "amazon-launch"
}

variable "region" {
  default = "us-central1"
}

variable "fastapi_image" {
  default = "us-central1-docker.pkg.dev/amazon-launch/amazon-predictor/app:fastapi"
}

variable "streamlit_image" {
  default = "us-central1-docker.pkg.dev/amazon-launch/amazon-predictor/app:streamlit"
}

# Look up project number dynamically — avoids hardcoding it
data "google_project" "project" {}

# Enable required APIs
resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudbuild" {
  service            = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

# Artifact Registry repository
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "amazon-predictor"
  format        = "DOCKER"
}

# Cloud Build service account permissions
# Cloud Build uses {project_number}@cloudbuild.gserviceaccount.com to build and push images

resource "google_project_iam_member" "cloudbuild_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# Compute Engine service account permissions
# Cloud Run uses {project_number}-compute@developer.gserviceaccount.com to pull images

resource "google_project_iam_member" "compute_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "compute_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

# FastAPI Cloud Run service
resource "google_cloud_run_v2_service" "fastapi" {
  name     = "fastapi-service"
  location = var.region

  template {
    containers {
      image = var.fastapi_image
      ports {
        container_port = 8000
      }
      resources {
        limits = {
          memory = "512Mi"
          cpu    = "1"
        }
      }
    }
  }
}

# Streamlit Cloud Run service
resource "google_cloud_run_v2_service" "streamlit" {
  name     = "streamlit-service"
  location = var.region

  template {
    containers {
      image = var.streamlit_image
      ports {
        container_port = 8501
      }
      resources {
        limits = {
          memory = "512Mi"
          cpu    = "1"
        }
      }
      env {
        name  = "API_URL"
        value = google_cloud_run_v2_service.fastapi.uri
      }
    }
  }
}

# Allow public access to FastAPI
resource "google_cloud_run_v2_service_iam_member" "fastapi_public" {
  name     = google_cloud_run_v2_service.fastapi.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Allow public access to Streamlit
resource "google_cloud_run_v2_service_iam_member" "streamlit_public" {
  name     = google_cloud_run_v2_service.streamlit.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Output the service URLs
output "fastapi_url" {
  value = google_cloud_run_v2_service.fastapi.uri
}

output "streamlit_url" {
  value = google_cloud_run_v2_service.streamlit.uri
}

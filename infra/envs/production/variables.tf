variable "aws_region" {
  type    = string
  default = "eu-central-1"
}

variable "remote_state_bucket" {
  description = "S3 bucket used for remote state"
  type        = string
  default     = "ml-project-terraform-state"
}

variable "networking_state_key" {
  description = "Path/key to networking state in S3 (e.g., networking/terraform.tfstate)"
  type        = string
  default     = "networking/terraform.tfstate"
}

variable "ecr_state_key" {
  description = "Path/key to ECR state in S3 (e.g., ecr/terraform.tfstate)"
  type        = string
  default     = "ecr/terraform.tfstate"
}

variable "project_name" {
  type    = string
  default = "ml-service"
}

variable "env" {
  type    = string
  default = "production"
}

variable "ecr_image_uri" {
  description = "ECR image URI including tag (e.g., 123456789012.dkr.ecr.eu-central-1.amazonaws.com/ml-service:staging-latest)"
  type        = string
  default     = "production-latest"
}

variable "container_name" {
  type    = string
  default = "ml-service"
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "alb_port" {
  type    = number
  default = 80
}

variable "ecs_task_cpu" {
  type    = number
  default = 512
}

variable "ecs_task_memory" {
  type    = number
  default = 1024
}

variable "task_container_cpu" {
  type    = number
  default = 512
}

variable "task_container_memory" {
  type    = number
  default = 1024
}

variable "desired_count" {
  type    = number
  default = 2
}

variable "cw_log_retention_days" {
  type    = number
  default = 14
}

variable "health_check_path" {
  type    = string
  default = "/health"
}

variable "health_check_matcher" {
  type    = string
  default = "200"
}

variable "health_check_interval" {
  type    = number
  default = 30
}

variable "health_check_timeout" {
  type    = number
  default = 5
}

variable "health_check_healthy_threshold" {
  type    = number
  default = 2
}

variable "health_check_unhealthy_threshold" {
  type    = number
  default = 2
}

variable "deployment_minimum_healthy_percent" {
  type    = number
  default = 50
}

variable "deployment_maximum_percent" {
  type    = number
  default = 200
}

variable "enable_container_insights" {
  type    = bool
  default = true
}

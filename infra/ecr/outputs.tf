output "repository_url" {
  value       = aws_ecr_repository.ml_service.repository_url
  description = "URL of the ECR repository"
}

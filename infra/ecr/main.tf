provider "aws" {
  region = var.aws_region
}

resource "aws_ecr_repository" "ml_service" {
  name                 = "ml-service"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = "shared"
    Project     = var.project_name
  }
}

terraform {
  backend "s3" {
    bucket         = "ml-project-terraform-state"
    key            = "ecr/terraform.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}

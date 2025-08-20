output "alb_dns_name" {
  value = module.alb.dns_name
}

output "ecs_cluster_id" {
  value = aws_ecs_cluster.this.id
}

output "ecs_service_name" {
  value = module.ecs_service.name
}

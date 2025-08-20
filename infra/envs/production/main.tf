terraform {
  required_version = ">= 1.3"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "terraform_remote_state" "networking" {
  backend = "s3"
  config = {
    bucket = var.remote_state_bucket
    key    = var.networking_state_key
    region = var.aws_region
  }
}

data "terraform_remote_state" "ecr" {
  backend = "s3"
  config = {
    bucket = var.remote_state_bucket
    key    = var.ecr_state_key
    region = var.aws_region
  }
}

# ------------------------------------------------------------------
# CloudWatch Log Group for ECS container logs
# ------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}-${var.env}"
  retention_in_days = var.cw_log_retention_days
  tags = {
    Project     = var.project_name
    Environment = var.env
  }
}

# ------------------------------------------------------------------
# ALB Security Group (allows inbound HTTP from internet)
# ------------------------------------------------------------------
module "alb_sg" {
  source  = "terraform-aws-modules/security-group/aws"
  # pin version if you want, e.g. version = "x.y.z"
  name        = "${var.project_name}-${var.env}-alb-sg"
  description = "ALB security group (allow HTTP in)"
  vpc_id      = data.terraform_remote_state.networking.outputs.vpc_id

  ingress_with_cidr_blocks = [
    {
      from_port   = var.alb_port
      to_port     = var.alb_port
      protocol    = "tcp"
      cidr_blocks = "0.0.0.0/0"
      description = "Allow HTTP from Internet"
    }
  ]

  egress_with_cidr_blocks = [
    {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = "0.0.0.0/0"
      description = "Allow all outbound"
    }
  ]

  tags = {
    Project     = var.project_name
    Environment = var.env
  }
}

# ------------------------------------------------------------------
# ECS tasks Security Group (allow inbound from ALB only)
# ------------------------------------------------------------------
module "ecs_sg" {
  source  = "terraform-aws-modules/security-group/aws"
  name        = "${var.project_name}-${var.env}-ecs-sg"
  description = "ECS tasks security group"
  vpc_id      = data.terraform_remote_state.networking.outputs.vpc_id

  ingress_with_source_security_group_id = [
    {
      from_port                       = var.container_port
      to_port                         = var.container_port
      protocol                        = "tcp"
      source_security_group_id        = module.alb_sg.security_group_id
      description                     = "Allow traffic from ALB"
    }
  ]

  egress_with_cidr_blocks = [
    {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = "0.0.0.0/0"
      description = "Allow all outbound (for NAT egress)"
    }
  ]

  tags = {
    Project     = var.project_name
    Environment = var.env
  }
}

# ------------------------------------------------------------------
# ALB
# ------------------------------------------------------------------
module "alb" {
  source  = "terraform-aws-modules/alb/aws"
  name    = "${var.project_name}-${var.env}-alb"
  enable_deletion_protection = false
  vpc_id  = data.terraform_remote_state.networking.outputs.vpc_id
  subnets = data.terraform_remote_state.networking.outputs.public_subnet_ids
  security_groups = [module.alb_sg.security_group_id]

  tags = {
    Project     = var.project_name
    Environment = var.env
  }
}

# ALB target group for ECS
resource "aws_lb_target_group" "ecs" {
  name        = "${var.project_name}-${var.env}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = data.terraform_remote_state.networking.outputs.vpc_id
  target_type = "ip"

  health_check {
    path                = var.health_check_path
    protocol            = "HTTP"
    matcher             = var.health_check_matcher
    interval            = var.health_check_interval
    timeout             = var.health_check_timeout
    healthy_threshold   = var.health_check_healthy_threshold
    unhealthy_threshold = var.health_check_unhealthy_threshold
  }

  tags = {
    Project     = var.project_name
    Environment = var.env
  }
}

# Listener, forwards the traffic to target group
resource "aws_lb_listener" "http" {
  load_balancer_arn = module.alb.arn
  port              = var.alb_port
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ecs.arn
  }
}

# ------------------------------------------------------------------
# ECS Cluster
# ------------------------------------------------------------------
resource "aws_ecs_cluster" "this" {
  name = "${var.project_name}-${var.env}-cluster"

  setting {
    name  = "containerInsights"
    value = var.enable_container_insights ? "enabled" : "disabled"
  }

  tags = {
    Project     = var.project_name
    Environment = var.env
  }
}

# ------------------------------------------------------------------
# ECS Service (Fargate) using module service submodule
# ------------------------------------------------------------------
module "ecs_service" {
  source  = "terraform-aws-modules/ecs/aws//modules/service"
  name        = "${var.project_name}-${var.env}-service"
  cluster_arn = aws_ecs_cluster.this.arn
  launch_type = "FARGATE"

  cpu    = var.ecs_task_cpu
  memory = var.ecs_task_memory


  # The module expects container_definitions as JSON string.
  # We'll supply a list (module accepts both map/list depending on module version) — here we use the `container_definitions` input as a JSON string of list.
  container_definitions = {
      (var.container_name) = {
      image     = "${data.terraform_remote_state.ecr.outputs.repository_url}:${var.ecr_image_uri}"
      cpu       = var.task_container_cpu
      memory    = var.task_container_memory
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "ENV", value = var.env },
        # add any other ENV vars you need here
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = var.project_name
        }
      }
    }
  }

  load_balancer = {
    service = {
      target_group_arn = aws_lb_target_group.ecs.arn
      container_name   = var.container_name
      container_port   = var.container_port
    }
  }

  subnet_ids         = data.terraform_remote_state.networking.outputs.private_subnet_ids
  security_group_ids = [module.ecs_sg.security_group_id]

  # Create the execution role (for pulling images + writing logs)
  create_task_exec_iam_role = true

  deployment_minimum_healthy_percent = var.deployment_minimum_healthy_percent
  deployment_maximum_percent         = var.deployment_maximum_percent

  enable_autoscaling       = true         # default, but be explicit
  autoscaling_min_capacity = 2            # start with 1 task
  autoscaling_max_capacity = 3            

  # Target-tracking policy example (CPU 60%)
  autoscaling_policies = {
    cpu_target = {
      policy_type = "TargetTrackingScaling"
      target_tracking_scaling_policy_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ECSServiceAverageCPUUtilization"
        }
        target_value = 60
      }
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.env
  }
}

# ------------------------------------------------------------------
# CloudWatch Log Group is created earlier (resource name aws_cloudwatch_log_group.ecs)
# But ECS's logConfiguration references it using its name (done above)
# ------------------------------------------------------------------
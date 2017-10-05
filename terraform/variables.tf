variable "aws_instance_type" {}

variable "aws_region" {
    type = "string"
    default = "eu-west-1"
}

variable "aws_availability_zone" {
    type = "string"
    default = "eu-west-1b"
}

variable "aws_access_key_id" {}
variable "aws_secret_access_key" {}

variable "fachung_volume_size" {}

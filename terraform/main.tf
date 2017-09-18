provider "aws" {
    profile = "personal"
    region = "eu-west-1"
    version = "~> 0.1"
}


resource "aws_security_group" "jupyter_notebook_sg" {
    name = "jupyter_notebook_sg"
    # Open up incoming ssh port
    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    # Open up incoming traffic to port 8888 used by Jupyter Notebook
    ingress {
        from_port   = 8888
        to_port     = 8888
        protocol    = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    # Open up outbound internet access
    egress {
        from_port   = 0
        to_port     = 0
        protocol    = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }
}


resource "aws_instance" "Fachung" {
    count = 1
    ami = "ami-a8d2d7ce"
    instance_type = "m4.xlarge"
    key_name = "fachung"
    tags {
        Name = "Fachung"
    }
    vpc_security_group_ids = ["${aws_security_group.jupyter_notebook_sg.id}"]

    provisioner "file" {
        source      = "configure.sh"
        destination = "/tmp/configure.sh"

        connection {
            type     = "ssh"
            user     = "ubuntu"
            private_key = "${file("~/.aws/fachung.pem")}"
        }
    }

    provisioner "file" {
        source      = "${file("~/.aws/fachung_credentials")}"
        destination = "/home/ubuntu/.aws/credentials"

        connection {
            type     = "ssh"
            user     = "ubuntu"
            private_key = "${file("~/.aws/fachung.pem")}"
        }
    }

    provisioner "remote-exec" {
        inline = [
            "chmod +x /tmp/configure.sh",
            "/tmp/configure.sh",
        ]
        connection {
            type     = "ssh"
            user     = "ubuntu"
            private_key = "${file("~/.aws/fachung.pem")}"
        }

    }

}


resource "aws_ebs_volume" "FachungData" {
  availability_zone = "eu-west-1"
  size              = 1
}


resource "aws_volume_attachment" "ebs_att" {
  device_name = "/dev/sdh"
  volume_id   = "${aws_ebs_volume.FachungData.id}"
  instance_id = "${aws_instance.Fachung.id}"
}


output "node_dns_name" {
    value = "${aws_instance.Node.public_dns}"
}

